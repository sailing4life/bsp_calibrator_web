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


def run(csv_path, seg_len_s=30, bsp_min=5.0, max_error_pct=15.0,
        max_bsp_std=1.5, max_hdg_std=13.0, max_heel_std=8.0, max_sog_std=1.5,
        twa_min=-170, twa_max=170):

    BSP = 'BSP'; SOG = 'SOG'; HDG = 'HDG'; COG = 'COG'
    HEEL = 'Heel'; TWA = 'TWA'; LAT = 'Lat'; LON = 'Lon'
    SEG_N = int(seg_len_s)

    plots = []
    diag = []

    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        return _err(f'Kan CSV niet lezen: {e}')

    for c in [BSP, SOG, HDG, COG, HEEL, TWA]:
        if c not in df_raw.columns:
            return _err(f'Kolom "{c}" niet gevonden in CSV. '
                        f'Beschikbare kolommen: {list(df_raw.columns[:10])}')

    # GPS track bewaren vóór filtering
    track = None
    if LAT in df_raw.columns and LON in df_raw.columns:
        t = df_raw[[LAT, LON]].copy()
        t[LAT] = pd.to_numeric(t[LAT], errors='coerce')
        t[LON] = pd.to_numeric(t[LON], errors='coerce')
        track = t.dropna()

    df = df_raw.copy()
    for c in [BSP, SOG, HDG, COG, HEEL, TWA]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    mask = df[BSP].notna() & df[SOG].notna() & df[HEEL].notna() & (df[BSP] != 0)
    df = df[mask].copy()
    if df.empty:
        return _err('Geen geldige data na basisfilter (BSP/SOG/Heel aanwezig, BSP ≠ 0).')

    n = len(df)
    diag.append(f'Rijen na basisfilter: {n}')

    df['BSP_STD'] = df[BSP].rolling(SEG_N, center=True).std()
    df['HDG_STD'] = df[HDG].rolling(SEG_N, center=True).std()
    df['HEEL_STD'] = df[HEEL].rolling(SEG_N, center=True).std()
    df['SOG_STD'] = df[SOG].rolling(SEG_N, center=True).std()

    stable = (
        (df['BSP_STD'] <= max_bsp_std) & (df['HDG_STD'] <= max_hdg_std) &
        (df['HEEL_STD'] <= max_heel_std) & (df['SOG_STD'] <= max_sog_std) &
        (df[BSP] >= bsp_min) & (df[TWA].between(twa_min, twa_max))
    )

    steps = [
        (f'BSP ≥ {bsp_min} kn',          df[BSP] >= bsp_min),
        (f'TWA [{twa_min}°, {twa_max}°]', df[TWA].between(twa_min, twa_max)),
        (f'BSP_STD ≤ {max_bsp_std} kn',  df['BSP_STD'] <= max_bsp_std),
        (f'SOG_STD ≤ {max_sog_std} kn',  df['SOG_STD'] <= max_sog_std),
        (f'HDG_STD ≤ {max_hdg_std}°',    df['HDG_STD'] <= max_hdg_std),
        (f'Heel_STD ≤ {max_heel_std}°',  df['HEEL_STD'] <= max_heel_std),
    ]
    cum = pd.Series([True] * n, index=df.index)
    for lbl, m in steps:
        cum &= m
        diag.append(f'{lbl}: {cum.sum()} ({100*cum.sum()/n:.1f}%)')
    diag.append(f'Stabiel totaal: {stable.sum()} ({100*stable.sum()/n:.1f}%)')

    # Stroom uit stabiele secties
    df_stab = df[stable & df[HDG].notna() & df[COG].notna()]
    if df_stab.empty:
        cur_x = cur_y = 0.0
    else:
        vw_x, vw_y = _vec(df_stab[BSP].values, df_stab[HDG].values)
        vg_x, vg_y = _vec(df_stab[SOG].values, df_stab[COG].values)
        cur_x = vg_x.mean() - vw_x.mean()
        cur_y = vg_y.mean() - vw_y.mean()

    cur_spd = np.sqrt(cur_x**2 + cur_y**2)
    cur_dir = np.degrees(np.arctan2(cur_x, cur_y)) % 360
    diag.append(f'Stroom (uit {len(df_stab)} stabiele samples): '
                f'{cur_spd:.3f} kn @ {cur_dir:.0f}°')

    # STW_true
    vg_x2, vg_y2 = _vec(df[SOG].values, df[COG].values)
    stw_x = vg_x2 - cur_x
    stw_y = vg_y2 - cur_y
    hdg_rad = np.deg2rad(df[HDG].values)
    df['STW_true'] = stw_x * np.sin(hdg_rad) + stw_y * np.cos(hdg_rad)

    # Stabiele blokken knippen
    block_id = (stable != stable.shift(fill_value=False)).cumsum()
    df['blk'] = block_id

    segs = []
    for bid, g in df[stable].groupby('blk'):
        g = g.sort_index()
        for i in range(0, len(g) - SEG_N + 1, SEG_N):
            s = g.iloc[i:i + SEG_N]
            if len(s) < SEG_N:
                continue
            segs.append({
                'BSP_mean':  s[BSP].mean(),
                'STW_mean':  s['STW_true'].mean(),
                'Heel_mean': s[HEEL].mean(),
                'Lat_mean':  s[LAT].mean() if LAT in s.columns else np.nan,
                'Lon_mean':  s[LON].mean() if LON in s.columns else np.nan,
            })

    if not segs:
        return _err('Geen stabiele secties gevonden. Pas drempelwaarden aan.', {'diag': diag})

    seg_df = pd.DataFrame(segs)
    seg_df['Pct'] = (seg_df['BSP_mean'] - seg_df['STW_mean']) / seg_df['BSP_mean'] * 100
    valid = seg_df[seg_df['Pct'].between(-max_error_pct, max_error_pct)].copy()
    diag.append(f'Secties (raw): {len(seg_df)},  na {max_error_pct}%-filter: {len(valid)}')

    if len(valid) < 5:
        return _err(f'Te weinig secties na filter ({len(valid)}). '
                    'Vergroot drempels of gebruik een langere log.', {'diag': diag})

    a_h, b_h, c_h = np.polyfit(valid['Heel_mean'].values, valid['Pct'].values, 2)
    a,   b,   c   = np.polyfit(valid['BSP_mean'].values,  valid['Pct'].values, 2)

    # Plot 1 — Heel vs fout%
    fig, ax = plt.subplots()
    x_h = valid['Heel_mean'].values
    ax.scatter(x_h, valid['Pct'].values, s=10, alpha=0.5, label='stabiele secties')
    xf = np.linspace(x_h.min(), x_h.max(), 200)
    ax.plot(xf, a_h*xf**2 + b_h*xf + c_h, lw=2, label='2e orde fit')
    ax.set_xlabel('Heel (deg)'); ax.set_ylabel('Fout [%]')
    ax.set_title('Heel vs BSP-fout (%)'); ax.grid(True); ax.legend()
    plt.tight_layout(); plots.append(_fig_to_b64(fig))

    # Plot 2 — BSP vs fout%
    fig, ax = plt.subplots()
    x_b = valid['BSP_mean'].values
    ax.scatter(x_b, valid['Pct'].values, s=15, alpha=0.5, label='stabiele secties')
    xf = np.linspace(x_b.min(), x_b.max(), 200)
    ax.plot(xf, a*xf**2 + b*xf + c, lw=2, label='2e orde fit')
    ax.axhline(0, ls='--', lw=1)
    ax.set_xlabel('BSP (kn)'); ax.set_ylabel('Fout [%]')
    ax.set_title('BSP vs fout (%)'); ax.grid(True); ax.legend()
    plt.tight_layout(); plots.append(_fig_to_b64(fig))

    # H5000 tabel
    heel_pts = np.array([-20., -10., 0., 10., 20.])
    spd_pts  = np.array([2.5, 5., 7.5, 10., 12.5, 15.])
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

    # Plot 4 — GPS track + stabiele secties + stroomvector
    if track is not None and not track.empty:
        geo = valid.dropna(subset=['Lat_mean', 'Lon_mean'])
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(track[LON], track[LAT], color='lightgray', lw=0.8, zorder=1, label='Track')
        if not geo.empty:
            sc = ax.scatter(geo['Lon_mean'], geo['Lat_mean'],
                            c=geo['Pct'], cmap='RdYlGn_r', s=40, zorder=2,
                            vmin=-max_error_pct, vmax=max_error_pct)
            plt.colorbar(sc, ax=ax, label='BSP fout [%]')
        lon_r = track[LON].max() - track[LON].min()
        lat_r = track[LAT].max() - track[LAT].min()
        sc_arr = max(lon_r, lat_r) * 0.15
        alx = track[LON].min() + lon_r * 0.08
        aly = track[LAT].max() - lat_r * 0.08
        ax.annotate('', xy=(alx + cur_x * sc_arr, aly + cur_y * sc_arr),
                    xytext=(alx, aly),
                    arrowprops=dict(arrowstyle='-|>', color='royalblue', lw=2), zorder=5)
        ax.text(alx + cur_x * sc_arr / 2, aly + cur_y * sc_arr / 2 - lat_r * 0.03,
                f'{cur_spd:.2f} kn @ {cur_dir:.0f}°',
                color='royalblue', fontsize=8, ha='center')
        ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
        ax.set_title('GPS track met stabiele secties')
        ax.set_aspect('equal'); ax.grid(True)
        plt.tight_layout(); plots.append(_fig_to_b64(fig))

    csv_b64 = base64.b64encode(h5.to_csv().encode()).decode()
    return {
        'error': None, 'plots': plots,
        'table_html': h5.to_html(classes='table table-sm table-bordered', border=0),
        'table_csv_b64': csv_b64,
        'stats': {'diag': diag, 'n_sections': len(valid)},
    }
