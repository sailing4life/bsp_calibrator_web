import html

import numpy as np
import pandas as pd


def _band(score):
    if score >= 75:
        return 'high'
    if score >= 50:
        return 'medium'
    return 'low'


def _label(score):
    if score >= 75:
        return 'hoog'
    if score >= 50:
        return 'bruikbaar'
    return 'laag'


def _cell_class(score):
    return f'confidence-{_band(score)}'


def _fmt_range(values):
    if len(values) == 0:
        return 'n/a'
    return f'{np.nanmin(values):.1f} - {np.nanmax(values):.1f}'


def build_confidence(valid, speed_col, heel_col, residuals, speed_points,
                     heel_points, max_error_pct):
    speeds = pd.to_numeric(valid[speed_col], errors='coerce').to_numpy(dtype=float)
    heels = pd.to_numeric(valid[heel_col], errors='coerce').to_numpy(dtype=float)
    keep = np.isfinite(speeds) & np.isfinite(heels)
    speeds = speeds[keep]
    heels = heels[keep]

    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else np.nan
    resid_scale = max(2.0, float(max_error_pct) * 0.45)
    residual_score = 1.0 if not np.isfinite(rmse) else 1 / (1 + (rmse / resid_scale) ** 2)

    if len(speeds) == 0:
        empty = pd.DataFrame(0, index=speed_points, columns=heel_points, dtype=int)
        return empty, {
            'score': 0,
            'band': 'low',
            'label': 'laag',
            'rmse_pct': rmse,
            'n': 0,
            'speed_range': 'n/a',
            'heel_range': 'n/a',
            'warnings': ['Geen geldige punten beschikbaar voor confidence scoring.'],
            'zero_support_cells': int(empty.size),
            'extrapolated_cells': int(empty.size),
        }

    speed_window = 1.25
    heel_window = 7.5
    speed_min, speed_max = float(np.min(speeds)), float(np.max(speeds))
    heel_min, heel_max = float(np.min(heels)), float(np.max(heels))

    scores = pd.DataFrame(index=speed_points, columns=heel_points, dtype=int)
    zero_support = 0
    extrapolated = 0

    for spd in speed_points:
        for heel in heel_points:
            local = ((np.abs(speeds - spd) <= speed_window) &
                     (np.abs(heels - heel) <= heel_window))
            local_count = int(local.sum())
            zero_support += int(local_count == 0)

            dist = np.sqrt(((speeds - spd) / speed_window) ** 2 +
                           ((heels - heel) / heel_window) ** 2)
            nearest_score = float(np.exp(-0.5 * np.min(dist)))
            count_score = min(1.0, local_count / 3.0)

            speed_out = max(speed_min - spd, 0, spd - speed_max) / speed_window
            heel_out = max(heel_min - heel, 0, heel - heel_max) / heel_window
            is_extrapolated = speed_out > 0 or heel_out > 0
            extrapolated += int(is_extrapolated)
            range_score = float(np.exp(-0.7 * (speed_out + heel_out)))

            score = 100 * (0.50 * range_score +
                           0.30 * nearest_score +
                           0.20 * count_score)
            score *= 0.75 + 0.25 * residual_score
            scores.loc[spd, heel] = int(np.clip(round(score), 0, 100))

    avg_score = float(scores.to_numpy(dtype=float).mean())
    warnings = []
    if extrapolated:
        warnings.append(f'{extrapolated} tabelcellen liggen buiten het gemeten bereik.')
    if zero_support:
        warnings.append(f'{zero_support} tabelcellen hebben geen nabijgelegen meetpunt.')

    neg = int((heels < -2).sum())
    pos = int((heels > 2).sum())
    if max(neg, pos) > 0 and min(neg, pos) / max(neg, pos) < 0.25:
        warnings.append(f'Heel-balans is zwak: {neg} negatief vs {pos} positief.')

    if np.isfinite(rmse) and rmse > resid_scale:
        warnings.append(f'Fit-spreiding is hoog ({rmse:.1f}%).')

    return scores, {
        'score': int(round(avg_score)),
        'band': _band(avg_score),
        'label': _label(avg_score),
        'rmse_pct': rmse,
        'n': int(len(speeds)),
        'speed_range': _fmt_range(speeds),
        'heel_range': _fmt_range(heels),
        'warnings': warnings,
        'zero_support_cells': zero_support,
        'extrapolated_cells': extrapolated,
    }


def confidence_table_html(scores):
    parts = [
        '<table class="table table-sm table-bordered confidence-table">',
        '<thead><tr><th>BSP (kn)</th>',
    ]
    for col in scores.columns:
        parts.append(f'<th>{html.escape(f"{float(col):g}")} deg</th>')
    parts.append('</tr></thead><tbody>')

    for idx, row in scores.iterrows():
        parts.append(f'<tr><th>{html.escape(f"{float(idx):g}")}</th>')
        for value in row:
            score = int(value)
            parts.append(
                f'<td class="confidence-cell {_cell_class(score)}">'
                f'<span class="confidence-score">{score}</span>'
                f'<span class="confidence-label">{_label(score)}</span>'
                '</td>'
            )
        parts.append('</tr>')

    parts.append('</tbody></table>')
    return ''.join(parts)
