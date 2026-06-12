import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def _vec(speed, angle_deg):
    rad = np.deg2rad(angle_deg)
    return speed * np.sin(rad), speed * np.cos(rad)


def _err(msg, stats=None):
    return {'error': msg, 'plots': [], 'table_html': '',
            'table_csv_b64': '', 'stats': stats or {'diag': []}}


def run(excel_path, sheet_name='Phase Table', bsp_min=3.0, max_error_pct=5.0,
        twa_min=-180, twa_max=180, leeway_k=10.0):

    BSP = 'BSP'; SOG = 'SOG'; HDG = 'HDG'; COG = 'COG'
    HEEL = 'HEEL'; TWA = 'TWA'; TIME = 'StartTime'; TACK = 'Tack'

    plots = []
    diag = []

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        return _err(f'Kan Excel niet lezen: {e}')

    for c in [BSP, SOG, HDG, COG, HEEL, TWA]:
        if c not in df.columns:
            return _err(f'Kolom "{c}" niet gevonden. '
                        f'Beschikbare kolommen: {list(df.columns[:15])}')
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # TWA naar [-180, 180] als de log 0..360 gebruikt
    if df[TWA].max() > 180:
        df[TWA] = ((df[TWA] + 180) % 360) - 180
        diag.append('TWA herschaald van 0..360 naar -180..180')

    if TIME in df.columns:
        df[TIME] = pd.to_datetime(df[TIME], errors='coerce')
        df['Date'] = df[TIME].dt.date
    else:
        df['Date'] = 'all'

    if TACK in df.columns:
        tack_side = (df[TACK].astype(str).str.split('/', expand=True)[0]
                     .map({'Port': +1, 'Stbd': -1}))
        df['Heel_signed'] = df[HEEL].abs() * tack_side
    else:
        df['Heel_signed'] = df[HEEL]

    mask = (df[BSP].notna() & df[SOG].notna() & df[HDG].notna() &
            df[COG].notna() & df[HEEL].notna() & (df[BSP] >= bsp_min))
    df = df[mask].copy()
    if df.empty:
        return _err('Geen geldige data na basisfilter.')

    diag.append(f'Fases na basisfilter: {len(df)}')

    # Stroom per dag
    cur_x_map = {}; cur_y_map = {}
    for date, g in df.groupby('Date'):
        gv = g[g[TWA].between(twa_min, twa_max)]
        if gv.empty:
            cur_x_map[date] = 0.0; cur_y_map[date] = 0.0
            continue
        # standaard leeway model: λ = k · heel / BSP²  (Heel_signed → tack volgt vanzelf)
        leeway = np.clip(leeway_k * gv['Heel_signed'].values
                         / np.maximum(gv[BSP].values, 1.0)**2, -15.0, 15.0)
        vw_x, vw_y = _vec(gv[BSP].values, gv[HDG].values + leeway)
        vg_x, vg_y = _vec(gv[SOG].values, gv[COG].values)
        cur_x_map[date] = vg_x.mean() - vw_x.mean()
        cur_y_map[date] = vg_y.mean() - vw_y.mean()
        spd = np.sqrt(cur_x_map[date]**2 + cur_y_map[date]**2)
        cdir = np.degrees(np.arctan2(cur_x_map[date], cur_y_map[date])) % 360
        diag.append(f'Stroom {date}: {spd:.3f} kn @ {cdir:.0f}°  (n={len(gv)})')

    df['CurX'] = df['Date'].map(cur_x_map)
    df['CurY'] = df['Date'].map(cur_y_map)

    vg_x2, vg_y2 = _vec(df[SOG].values, df[COG].values)
    stw_x = vg_x2 - df['CurX'].values
    stw_y = vg_y2 - df['CurY'].values
    hdg_rad = np.deg2rad(df[HDG].values)
    df['STW_true'] = stw_x * np.sin(hdg_rad) + stw_y * np.cos(hdg_rad)

    df['Pct'] = (df[BSP] - df['STW_true']) / df[BSP] * 100
    valid = df[df['Pct'].between(-max_error_pct, max_error_pct)].copy()
    diag.append(f'Fases (raw): {len(df)},  na {max_error_pct}%-filter: {len(valid)}')

    if len(valid) < 5:
        return _err(f'Te weinig data na filter ({len(valid)}).', {'diag': diag})

    x_h = valid['Heel_signed'].values
    x_b = valid[BSP].values
    y   = valid['Pct'].values

    # Sequentiële fit: eerst snelheid, dan heel op de residuen.
    # Onafhankelijk fitten telt dubbel omdat heel en BSP gecorreleerd zijn.
    a, b, c = np.polyfit(x_b, y, 2)
    resid = y - (a*x_b**2 + b*x_b + c)
    a_h, b_h, c_h = np.polyfit(x_h, resid, 2)

    # Plot 1 — Heel_signed vs restfout%
    fig, ax = plt.subplots()
    ax.scatter(x_h, resid, s=10, alpha=0.5, label='fases')
    xf = np.linspace(x_h.min(), x_h.max(), 200)
    ax.plot(xf, a_h*xf**2 + b_h*xf + c_h, lw=2, label='2e orde fit')
    ax.set_xlabel('Heel_signed (deg)  [Port +, Stbd −]')
    ax.set_ylabel('Restfout [%] (na BSP-fit)')
    ax.set_title('Heel_signed vs BSP-restfout (%)')
    ax.grid(True); ax.legend(); plt.tight_layout(); plots.append(_fig_to_b64(fig))

    # Plot 2 — BSP vs fout%
    fig, ax = plt.subplots()
    ax.scatter(x_b, y, s=15, alpha=0.5, label='fases')
    xf = np.linspace(x_b.min(), x_b.max(), 200)
    ax.plot(xf, a*xf**2 + b*xf + c, lw=2, label='2e orde fit')
    ax.axhline(0, ls='--', lw=1)
    ax.set_xlabel('BSP (kn)'); ax.set_ylabel('Fout [%]')
    ax.set_title('BSP vs fout (%)'); ax.grid(True); ax.legend()
    plt.tight_layout(); plots.append(_fig_to_b64(fig))

    # H5000 tabel
    heel_pts = np.array([-20., -10., 0., 10., 20.])
    spd_pts  = np.array([2.5, 5., 7.5, 10., 12.5, 15.])

    if x_h.min() > heel_pts.min() or x_h.max() < heel_pts.max() \
            or x_b.min() > spd_pts.min() or x_b.max() < spd_pts.max():
        diag.append(f'⚠ Tabel geëxtrapoleerd buiten databereik: '
                    f'Heel data [{x_h.min():.0f}°, {x_h.max():.0f}°], '
                    f'BSP data [{x_b.min():.1f}, {x_b.max():.1f}] kn — '
                    f'cellen daarbuiten zijn onbetrouwbaar')

    h5 = pd.DataFrame(index=spd_pts, columns=heel_pts, dtype=float)
    for i, spd in enumerate(spd_pts):
        heel_corr = -(a_h * heel_pts**2 + b_h * heel_pts)
        base_corr = -spd * (a * spd**2 + b * spd + c) / 100
        h5.iloc[i] = spd * heel_corr / 100 + base_corr
    h5 = h5.round(3)
    h5.index.name = 'BSP (kn)'; h5.columns.name = 'Heel (deg)'

    # Plot 3 — H5000 lijnen
    fig, ax = plt.subplots()
    for spd in h5.index:
        ax.plot(h5.columns, h5.loc[spd], marker='o', label=f'{spd} kn')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Heel (deg)'); ax.set_ylabel('Correctie (kn)')
    ax.set_title('H5000 BSP Heel correctie tabel')
    ax.legend(title='BSP'); ax.grid(True)
    plt.tight_layout(); plots.append(_fig_to_b64(fig))

    csv_b64 = base64.b64encode(h5.to_csv().encode()).decode()
    return {
        'error': None, 'plots': plots,
        'table_html': h5.to_html(classes='table table-sm table-bordered', border=0),
        'table_csv_b64': csv_b64,
        'stats': {'diag': diag, 'n_sections': len(valid)},
    }
