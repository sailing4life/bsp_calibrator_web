import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from confidence import build_confidence, confidence_table_html


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
            'table_csv_b64': '', 'confidence': None,
            'stats': stats or {'diag': []}}


def _circ_std_deg(series_deg, window):
    """Rolling circulaire standaarddeviatie (graden), wrap-safe rond 0/360."""
    rad = np.deg2rad(series_deg)
    s = np.sin(rad).rolling(window, center=True).mean()
    c = np.cos(rad).rolling(window, center=True).mean()
    R = np.sqrt(s**2 + c**2).clip(upper=1.0)
    return np.degrees(np.sqrt(-2 * np.log(R.clip(lower=1e-12))))


def _add_basemap(ax, lon_min, lon_max, lat_min, lat_max):
    """OSM-tegels als kaartachtergrond; faalt stil zonder internet."""
    import math, urllib.request
    from PIL import Image

    def lon2x(lon, z): return (lon + 180) / 360 * 2**z
    def lat2y(lat, z):
        return (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**z
    def y2lat(y, z):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2**z))))

    try:
        for z in range(16, 0, -1):
            x0, x1 = int(lon2x(lon_min, z)), int(lon2x(lon_max, z))
            y0, y1 = int(lat2y(lat_max, z)), int(lat2y(lat_min, z))
            if (x1 - x0 + 1) * (y1 - y0 + 1) <= 12:
                break
        img = Image.new('RGB', ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256))
        for xt in range(x0, x1 + 1):
            for yt in range(y0, y1 + 1):
                url = f'https://tile.openstreetmap.org/{z}/{xt}/{yt}.png'
                req = urllib.request.Request(
                    url, headers={'User-Agent': 'bsp-calibrator/1.0'})
                with urllib.request.urlopen(req, timeout=4) as r:
                    tile = Image.open(io.BytesIO(r.read())).convert('RGB')
                img.paste(tile, ((xt - x0) * 256, (yt - y0) * 256))
        extent = [x0 / 2**z * 360 - 180, (x1 + 1) / 2**z * 360 - 180,
                  y2lat(y1 + 1, z), y2lat(y0, z)]
        ax.imshow(img, extent=extent, zorder=0, interpolation='bilinear', alpha=0.7)
        ax.text(0.99, 0.01, '© OpenStreetMap', transform=ax.transAxes,
                fontsize=6, ha='right', va='bottom', color='gray')
        return True
    except Exception:
        return False


def run(csv_path, seg_len_s=30, bsp_min=5.0, max_error_pct=15.0,
        max_bsp_std=1.5, max_hdg_std=13.0, max_heel_std=8.0, max_sog_std=1.5,
        twa_min=-170, twa_max=170, leeway_k=0.0, fit_mode='independent'):

    BSP = 'BSP'; SOG = 'SOG'; HDG = 'HDG'; COG = 'COG'
    HEEL = 'Heel'; TWA = 'TWA'; LAT = 'Lat'; LON = 'Lon'

    plots = []
    diag = []

    try:
        df_raw = pd.read_csv(csv_path)
    except Exception as e:
        return _err(f'Kan CSV niet lezen: {e}')

    # Sample frequentie detecteren uit tijdkolom
    _time_cols = [c for c in df_raw.columns
                  if any(k in c.lower() for k in ('time', 'date', 'tijd', 'ts', 'utc'))]
    freq_hz = None
    for _tc in _time_cols:
        try:
            _ts = pd.to_datetime(df_raw[_tc], errors='coerce').dropna()
            if len(_ts) > 10:
                _dt = _ts.diff().dt.total_seconds().median()
                if 0.05 < _dt < 60:
                    freq_hz = 1.0 / _dt
                    diag.append(f'Sample frequentie: {freq_hz:.2f} Hz '
                                f'(interval {_dt:.2f}s, kolom "{_tc}")')
                    break
        except Exception:
            pass
    if freq_hz is None:
        diag.append('Sample frequentie: onbekend (geen tijdkolom herkend), '
                    '1 Hz aangenomen')
        SEG_N = int(seg_len_s)
    else:
        SEG_N = max(2, int(round(seg_len_s * freq_hz)))
    diag.append(f'Sectie lengte: {seg_len_s}s = {SEG_N} samples')

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

    # TWA naar [-180, 180] als de log 0..360 gebruikt
    if df[TWA].max() > 180:
        df[TWA] = ((df[TWA] + 180) % 360) - 180
        diag.append('TWA herschaald van 0..360 naar -180..180')

    mask = df[BSP].notna() & df[SOG].notna() & df[HEEL].notna() & (df[BSP] != 0)
    df = df[mask].copy()
    if df.empty:
        return _err('Geen geldige data na basisfilter (BSP/SOG/Heel aanwezig, BSP ≠ 0).')

    n = len(df)
    diag.append(f'Rijen na basisfilter: {n}')
    if abs(leeway_k) < 1e-9:
        diag.append('Leeway-correctie: uit')
    else:
        diag.append(f'Leeway-correctie: aan (k={leeway_k:g})')

    df['BSP_STD'] = df[BSP].rolling(SEG_N, center=True).std()
    df['HDG_STD'] = _circ_std_deg(df[HDG], SEG_N)
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

    # Stroom uit stabiele secties (met leeway-correctie op de watervector)
    df_stab = df[stable & df[HDG].notna() & df[COG].notna()]

    def _current(sub):
        if sub.empty:
            return 0.0, 0.0
        # Standaard leeway model: grootte λ = k · |heel| / BSP², richting altijd
        # naar lij. De tack (en dus de lijzijde) leiden we af uit sign(TWA),
        # niet uit de ruwe heel-conventie die per logsysteem verschilt:
        #   TWA>0 = stuurboordtack → lij = bakboord → HDG − λ
        #   TWA<0 = bakboordtack   → lij = stuurboord → HDG + λ
        mag = np.clip(leeway_k * np.abs(sub[HEEL].values)
                      / np.maximum(sub[BSP].values, 1.0)**2, 0.0, 10.0)
        leeway = -np.sign(sub[TWA].values) * mag
        vw_x, vw_y = _vec(sub[BSP].values, sub[HDG].values + leeway)
        vg_x, vg_y = _vec(sub[SOG].values, sub[COG].values)
        return vg_x.mean() - vw_x.mean(), vg_y.mean() - vw_y.mean()

    cur_x, cur_y = _current(df_stab)
    cur_spd = np.sqrt(cur_x**2 + cur_y**2)
    cur_dir = np.degrees(np.arctan2(cur_x, cur_y)) % 360
    diag.append(f'Stroom (uit {len(df_stab)} stabiele samples, '
                f'leeway k={leeway_k:g}): {cur_spd:.3f} kn @ {cur_dir:.0f}°')

    # Per-tack sanity check: groot verschil duidt op leeway-/kalibratie-bias
    tack_cur = {}
    for lbl, sub in [('Stbd (TWA>0)', df_stab[df_stab[TWA] > 0]),
                     ('Port (TWA<0)', df_stab[df_stab[TWA] < 0])]:
        if len(sub) >= 10:
            tx, ty = _current(sub)
            tack_cur[lbl] = (tx, ty)
            tspd = np.sqrt(tx**2 + ty**2)
            tdir = np.degrees(np.arctan2(tx, ty)) % 360
            diag.append(f'  Stroom {lbl}: {tspd:.3f} kn @ {tdir:.0f}°  (n={len(sub)})')
    if len(tack_cur) == 2:
        (sx, sy), (px, py) = tack_cur.values()
        dmag = np.sqrt((sx - px)**2 + (sy - py)**2)
        diag.append(f'  Verschil Stbd-Port: {dmag:.3f} kn'
                    + ('  ⚠ check leeway/kalibratie' if dmag > 0.3 else ''))

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

    x_h = valid['Heel_mean'].values
    x_b = valid['BSP_mean'].values
    y   = valid['Pct'].values

    a, b, c = np.polyfit(x_b, y, 2)
    speed_fit = a*x_b**2 + b*x_b + c
    if fit_mode == 'sequential':
        # Heel gefit op de residuen van de BSP-fit (voorkomt dubbeltelling
        # bij scheve data, maar ruisiger bij gebalanceerde data).
        y_h = y - speed_fit
        heel_lbl = 'Restfout [%] (na BSP-fit)'
    else:
        # Onafhankelijke fit (origineel) — robuuster bij gebalanceerde data.
        y_h = y
        heel_lbl = 'Fout [%]'
    a_h, b_h, c_h = np.polyfit(x_h, y_h, 2)
    heel_fit = a_h*x_h**2 + b_h*x_h + c_h
    if fit_mode == 'sequential':
        fit_residuals = y_h - heel_fit
    else:
        fit_residuals = np.concatenate([y - speed_fit, y_h - heel_fit])
    diag.append(f'Fit-methode: {fit_mode}')

    # Plot 1 — Heel vs fout%
    fig, ax = plt.subplots()
    ax.scatter(x_h, y_h, s=10, alpha=0.5, label='stabiele secties')
    xf = np.linspace(x_h.min(), x_h.max(), 200)
    ax.plot(xf, a_h*xf**2 + b_h*xf + c_h, lw=2, label='2e orde fit')
    ax.set_xlabel('Heel (deg)'); ax.set_ylabel(heel_lbl)
    ax.set_title('Heel vs BSP-fout (%)'); ax.grid(True); ax.legend()
    plt.tight_layout(); plots.append(_fig_to_b64(fig))

    # Plot 2 — BSP vs fout%
    fig, ax = plt.subplots()
    ax.scatter(x_b, y, s=15, alpha=0.5, label='stabiele secties')
    xf = np.linspace(x_b.min(), x_b.max(), 200)
    ax.plot(xf, a*xf**2 + b*xf + c, lw=2, label='2e orde fit')
    ax.axhline(0, ls='--', lw=1)
    ax.set_xlabel('BSP (kn)'); ax.set_ylabel('Fout [%]')
    ax.set_title('BSP vs fout (%)'); ax.grid(True); ax.legend()
    plt.tight_layout(); plots.append(_fig_to_b64(fig))

    # H5000 tabel
    heel_pts = np.array([-20., -10., 0., 10., 20.])
    spd_pts  = np.array([2.5, 5., 7.5, 10., 12.5, 15.])

    heel_lo, heel_hi = x_h.min(), x_h.max()
    bsp_lo, bsp_hi = valid['BSP_mean'].min(), valid['BSP_mean'].max()
    if heel_lo > heel_pts.min() or heel_hi < heel_pts.max() \
            or bsp_lo > spd_pts.min() or bsp_hi < spd_pts.max():
        diag.append(f'⚠ Tabel geëxtrapoleerd buiten databereik: '
                    f'Heel data [{heel_lo:.0f}°, {heel_hi:.0f}°], '
                    f'BSP data [{bsp_lo:.1f}, {bsp_hi:.1f}] kn — '
                    f'cellen daarbuiten zijn onbetrouwbaar')

    h5 = pd.DataFrame(index=spd_pts, columns=heel_pts, dtype=float)
    for i, spd in enumerate(spd_pts):
        heel_corr = -(a_h * heel_pts**2 + b_h * heel_pts)
        base_corr = -spd * (a * spd**2 + b * spd + c) / 100
        h5.iloc[i] = spd * heel_corr / 100 + base_corr
    h5 = h5.round(3)
    h5.index.name = 'BSP (kn)'; h5.columns.name = 'Heel (deg)'

    confidence_scores, confidence_summary = build_confidence(
        valid, 'BSP_mean', 'Heel_mean', fit_residuals, spd_pts, heel_pts,
        max_error_pct)
    confidence_html = confidence_table_html(confidence_scores)
    rmse = confidence_summary['rmse_pct']
    rmse_txt = f'{rmse:.1f}%' if np.isfinite(rmse) else 'n/a'
    diag.append(f'Confidence gemiddeld: {confidence_summary["score"]}/100 '
                f'({confidence_summary["label"]}), fit-spreiding {rmse_txt}')
    for warning in confidence_summary['warnings']:
        diag.append(f'Confidence waarschuwing: {warning}')

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

        cosl = np.cos(np.deg2rad(track[LAT].mean()))
        lon_r = track[LON].max() - track[LON].min()
        lat_r = track[LAT].max() - track[LAT].min()
        pad_lon = max(lon_r, 0.001) * 0.10
        pad_lat = max(lat_r, 0.001) * 0.10
        _add_basemap(ax,
                     track[LON].min() - pad_lon, track[LON].max() + pad_lon,
                     track[LAT].min() - pad_lat, track[LAT].max() + pad_lat)

        ax.plot(track[LON], track[LAT], color='dimgray', lw=0.8, zorder=1, label='Track')
        if not geo.empty:
            sc = ax.scatter(geo['Lon_mean'], geo['Lat_mean'],
                            c=geo['Pct'], cmap='RdYlGn_r', s=40, zorder=2,
                            vmin=-max_error_pct, vmax=max_error_pct)
            plt.colorbar(sc, ax=ax, label='BSP fout [%]')
        # oost-component door cos(lat) delen zodat de pijlrichting op de kaart klopt
        sc_arr = max(lon_r, lat_r) * 0.15
        alx = track[LON].min() + lon_r * 0.08
        aly = track[LAT].max() - lat_r * 0.08
        ax.annotate('', xy=(alx + cur_x * sc_arr / cosl, aly + cur_y * sc_arr),
                    xytext=(alx, aly),
                    arrowprops=dict(arrowstyle='-|>', color='royalblue', lw=2), zorder=5)
        ax.text(alx + cur_x * sc_arr / cosl / 2,
                aly + cur_y * sc_arr / 2 - lat_r * 0.03,
                f'{cur_spd:.2f} kn @ {cur_dir:.0f}°',
                color='royalblue', fontsize=8, ha='center', zorder=5)
        ax.set_xlim(track[LON].min() - pad_lon, track[LON].max() + pad_lon)
        ax.set_ylim(track[LAT].min() - pad_lat, track[LAT].max() + pad_lat)
        ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
        ax.set_title('GPS track met stabiele secties')
        ax.set_aspect(1 / cosl); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plots.append(_fig_to_b64(fig))

    csv_b64 = base64.b64encode(h5.to_csv().encode()).decode()
    return {
        'error': None, 'plots': plots,
        'table_html': h5.to_html(classes='table table-sm table-bordered', border=0),
        'table_csv_b64': csv_b64,
        'confidence': {
            'table_html': confidence_html,
            'summary': confidence_summary,
        },
        'stats': {'diag': diag, 'n_sections': len(valid)},
    }
