import os
import tempfile
import traceback
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB


def _err_result(msg):
    return {'error': msg, 'plots': [], 'table_html': '',
            'table_csv_b64': '', 'stats': {'diag': []}}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run/1min', methods=['POST'])
def run_1min():
    from cal_1min import run
    f = request.files.get('csvfile')
    if not f or f.filename == '':
        return redirect(url_for('index'))

    fd, tmp = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    try:
        f.save(tmp)
        result = run(
            tmp,
            seg_len_s=int(request.form.get('seg_len_s', 30)),
            bsp_min=float(request.form.get('bsp_min', 5.0)),
            max_bsp_std=float(request.form.get('max_bsp_std', 1.5)),
            max_sog_std=float(request.form.get('max_sog_std', 1.5)),
            max_hdg_std=float(request.form.get('max_hdg_std', 13.0)),
            max_heel_std=float(request.form.get('max_heel_std', 8.0)),
            max_error_pct=float(request.form.get('max_error_pct', 15.0)),
            leeway_k=float(request.form.get('leeway_k', 10.0)),
        )
    except Exception:
        result = _err_result(f'Verwerking mislukt:\n{traceback.format_exc()}')
    finally:
        os.unlink(tmp)

    return render_template('result.html', result=result, tool='CSV Logfile')


@app.route('/run/phasetable', methods=['POST'])
def run_phasetable():
    from cal_phasetable import run
    f = request.files.get('xlsfile')
    if not f or f.filename == '':
        return redirect(url_for('index'))

    fd, tmp = tempfile.mkstemp(suffix='.xlsx')
    os.close(fd)
    try:
        f.save(tmp)
        result = run(
            tmp,
            sheet_name=request.form.get('sheet_name', 'Phase Table'),
            bsp_min=float(request.form.get('bsp_min', 3.0)),
            max_error_pct=float(request.form.get('max_error_pct', 5.0)),
            leeway_k=float(request.form.get('leeway_k', 10.0)),
        )
    except Exception:
        result = _err_result(f'Verwerking mislukt:\n{traceback.format_exc()}')
    finally:
        os.unlink(tmp)

    return render_template('result.html', result=result, tool='PhaseTable Excel')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
