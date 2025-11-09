from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response, jsonify, g
import os
import glob
import json
from datetime import datetime
import time
from werkzeug.utils import secure_filename
from onnx_infer import ONNXYoloDetector
import sys
import threading
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import traceback
import uuid
from werkzeug.exceptions import HTTPException
from typing import Optional
import hashlib
import numpy as np
import shutil
try:
    from db import init_db, setup_database_from_env, SessionLocal, UploadHistory, connect_with_params
except Exception:
    init_db = None  
    setup_database_from_env = None  
    SessionLocal = None  
    UploadHistory = None  
    connect_with_params = None  
try:
    import cv2
except Exception as _e:
    cv2 = None  
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
HISTORY_FILE = os.path.join(UPLOAD_FOLDER, 'history.json')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
DEFAULT_MODEL = os.path.join(BASE_DIR, 'models', 'best_9.onnx')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
try:
    import redis
except Exception:
    redis = None
REDIS_URL = os.getenv('REDIS_URL')
_redis_client = None
if redis is not None and REDIS_URL:
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        app.logger.info(f"Redis 缓存已连接: {REDIS_URL}")
    except Exception as e:
        app.logger.warning(f"无法连接 Redis ({REDIS_URL}): {e}")
def _setup_logging():
    logger = app.logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, 'app.log'), maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        logger.addHandler(console)
_setup_logging()
DB_ENABLED = False
try:
    from db_init import attempt_database_setup
    DB_ENABLED, _msg = attempt_database_setup(connect_with_params, init_db, app.logger)
except Exception as e:
    app.logger.warning(f"数据库初始化失败，继续以文件模式运行: {e}")
_model_cache = {}
@app.before_request
def _before_request_logging():
    g._started_at = time.time()
    g._req_id = uuid.uuid4().hex[:8]
    try:
        content_type = request.headers.get('Content-Type', '')
        if request.method in ('POST', 'PUT', 'PATCH') and 'multipart/form-data' not in content_type:
            body_preview = request.get_data(cache=True, as_text=True)
            if body_preview and len(body_preview) > 1000:
                body_preview = body_preview[:1000] + '…(truncated)'
        else:
            body_preview = None
    except Exception:
        body_preview = None
    app.logger.info(f"[{g._req_id}] -> {request.method} {request.path} ip={request.remote_addr} ua={request.user_agent.string}")
    if body_preview:
        app.logger.debug(f"[{g._req_id}] body: {body_preview}")
@app.after_request
def _after_request_logging(response):
    try:
        dur_ms = int((time.time() - getattr(g, '_started_at', time.time())) * 1000)
        req_id = getattr(g, '_req_id', '-')
        app.logger.info(f"[{req_id}] <- {response.status_code} {request.method} {request.path} {dur_ms}ms")
    except Exception:
        pass
    return response
def _wants_json() -> bool:
    try:
        if request.path.startswith('/api/'):
            return True
        accept = request.headers.get('Accept', '')
        return 'application/json' in accept
    except Exception:
        return False
@app.errorhandler(404)
def handle_404(err):
    msg = {'ok': False, 'error': 'Not Found', 'path': request.path}
    if _wants_json():
        return jsonify(msg), 404
    return f"404 Not Found: {request.path}", 404
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        code = e.code or 500
        desc = getattr(e, 'description', str(e))
        app.logger.warning(f"HTTPException {code} at {request.path}: {desc}")
        if _wants_json():
            return jsonify({'ok': False, 'error': desc, 'code': code}), code
        return f"Error {code}: {desc}", code
    tb = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    app.logger.error(f"Unhandled exception at {request.method} {request.path}: {e}\n{tb}")
    if _wants_json():
        return jsonify({'ok': False, 'error': 'Internal Server Error'}), 500
    return '500 Internal Server Error', 500
def _tail_file_lines(path: str, max_lines: int = 400) -> list[str]:
    max_lines = max(1, min(5000, int(max_lines)))
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = -1
            data = b''
            nl = 0
            while size > 0 and nl <= max_lines:
                step = 1024
                pos = max(0, size - step)
                f.seek(pos)
                chunk = f.read(size - pos)
                data = chunk + data
                nl = data.count(b'\n')
                size = pos
            text = data.decode('utf-8', errors='replace')
            lines = text.splitlines()
            return lines[-max_lines:]
    except FileNotFoundError:
        return []
@app.route('/api/app/logs')
def api_app_logs():
    try:
        tail = int(request.args.get('tail', '400'))
    except Exception:
        tail = 400
    tail = max(1, min(5000, tail))
    file_path = os.path.join(LOG_DIR, 'app.log')
    if not os.path.isfile(file_path):
        return jsonify({'ok': True, 'lines': [], 'count': 0, 'file': {'path': file_path, 'exists': False}})
    lines = _tail_file_lines(file_path, tail)
    try:
        st = os.stat(file_path)
        meta = {
            'path': file_path,
            'exists': True,
            'size': st.st_size,
            'mtime': datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        }
    except Exception:
        meta = {'path': file_path, 'exists': True}
    return jsonify({'ok': True, 'lines': lines, 'count': len(lines), 'file': meta})
def discover_models():
    cache_key = 'models:discover'
    if _redis_client:
        try:
            cached = _redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass
    patterns = [os.path.join(BASE_DIR, 'models', '**', '*.onnx')]
    found = set()
    for pattern in patterns:
        for p in glob.glob(pattern, recursive=True):
            if os.path.isfile(p):
                found.add(os.path.abspath(p))
    models = sorted(found)
    model_items = [
        {
            'path': m,
            'label': os.path.relpath(m, BASE_DIR) if m.startswith(BASE_DIR) else os.path.basename(m),
        }
        for m in models
    ]
    if _redis_client:
        try:
            _redis_client.set(cache_key, json.dumps(model_items), ex=30)
        except Exception:
            pass
    return model_items
def _parse_names_from_yaml(yaml_path: str) -> list[str] | None:
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except Exception:
        return None
    names: list[str] | None = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('names:'):
            after = s[len('names:'):].strip()
            if after.startswith('['):
                try:
                    content = after
                    try:
                        import json as _json
                        arr = _json.loads(content)
                    except Exception:
                        import ast as _ast
                        arr = _ast.literal_eval(content)
                    if isinstance(arr, list):
                        names = [str(x) for x in arr]
                        break
                except Exception:
                    pass
            elif after.startswith('{'):
                try:
                    import ast as _ast
                    obj = _ast.literal_eval(after)
                    if isinstance(obj, dict) and obj:
                        items = sorted(((int(k), v) for k, v in obj.items()), key=lambda x: x[0])
                        max_k = items[-1][0]
                        buf = [None] * (max_k + 1)
                        for k, v in items:
                            if 0 <= k <= max_k:
                                buf[k] = str(v)
                        names = [b if b is not None else str(i) for i, b in enumerate(buf)]
                        break
                except Exception:
                    pass
            else:
                vals: list[str] = []
                j = i + 1
                while j < len(lines):
                    t = lines[j]
                    if t.strip().startswith('- '):
                        item = t.split('- ', 1)[1].strip().strip('"').strip("'")
                        vals.append(item)
                        j += 1
                    elif ':' in t and not t.strip().startswith('- '):
                        try:
                            key, val = t.split(':', 1)
                            key = key.strip()
                            val = val.strip().strip('"').strip("'")
                            if key.isdigit():
                                if names is None:
                                    names = []
                                idx = int(key)
                                while len(names) <= idx:
                                    names.append(str(len(names)))
                                names[idx] = val
                            j += 1
                            continue
                        except Exception:
                            pass
                    elif t.startswith(' ') or t.startswith('\t'):
                        j += 1
                    else:
                        break
                if vals:
                    names = vals
                if names:
                    break
    return names
def get_default_class_names() -> list[str] | None:
    cand = os.path.abspath(os.path.join(BASE_DIR, '..', 'Data', 'data.yaml'))
    if os.path.isfile(cand):
        return _parse_names_from_yaml(cand)
    return None
def get_model(model_path: str):
    return get_model_with_device(model_path, device=None)
def _providers_for_device(device: str | None) -> list[str]:
    try:
        import onnxruntime as _ort
        avail = set(_ort.get_available_providers() or [])
    except Exception:
        avail = set()
    if not device or device == 'auto':
        if 'CUDAExecutionProvider' in avail:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if 'MPSExecutionProvider' in avail:
            return ['MPSExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']
    d = device.lower()
    if d == 'cpu':
        return ['CPUExecutionProvider']
    if d.startswith('cuda'):
        if 'CUDAExecutionProvider' in avail:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']
    if d == 'mps':
        if 'MPSExecutionProvider' in avail:
            return ['MPSExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']
    return ['CPUExecutionProvider']
def get_model_with_device(model_path: str, device: str | None = None):
    abspath = os.path.abspath(model_path)
    key = (abspath, device or 'auto')
    if key in _model_cache:
        return _model_cache[key]
    providers = _providers_for_device(device)
    try:
        model = ONNXYoloDetector(abspath, providers=providers)
    except Exception:
        model = ONNXYoloDetector(abspath)
    _model_cache[key] = model
    return model
def _model_label(path: str) -> str:
    return os.path.relpath(path, BASE_DIR) if path.startswith(BASE_DIR) else os.path.basename(path)
def load_history() -> list:
    if DB_ENABLED and SessionLocal is not None and UploadHistory is not None:
        try:
            with SessionLocal() as s:  
                rows = s.query(UploadHistory).order_by(UploadHistory.id.desc()).all() 
                return [r.to_dict() for r in rows]
        except Exception as e:
            app.logger.error(f"读取数据库历史失败，回退到文件: {e}")
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []
def save_history(items: list):
    if DB_ENABLED and SessionLocal is not None and UploadHistory is not None:
        try:
            with SessionLocal() as s:  
                s.query(UploadHistory).delete() 
                for it in items:
                    t_str = it.get('time')
                    try:
                        t = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S') if t_str else datetime.now()
                    except Exception:
                        t = datetime.now()
                    row = UploadHistory(  
                        time=t,
                        model_path=it.get('model_path'),
                        model_label=it.get('model_label'),
                        filename=it.get('filename'),
                        result_filename=it.get('result_filename'),
                        labels=json.dumps(it.get('labels')) if it.get('labels') is not None else None,
                    )
                    s.add(row)
                s.commit()
                return
        except Exception as e:
            app.logger.error(f"写入数据库历史失败，回退到文件: {e}")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
def add_history_entry(entry: dict):
    if DB_ENABLED and SessionLocal is not None and UploadHistory is not None:
        try:
            with SessionLocal() as s:  
                t_str = entry.get('time')
                try:
                    t = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S') if t_str else datetime.now()
                except Exception:
                    t = datetime.now()
                row = UploadHistory( 
                    time=t,
                    model_path=entry.get('model_path'),
                    model_label=entry.get('model_label'),
                    filename=entry.get('filename'),
                    result_filename=entry.get('result_filename'),
                    labels=json.dumps(entry.get('labels')) if entry.get('labels') is not None else None,
                )
                s.add(row)
                s.commit()
                return
        except Exception as e:
            app.logger.error(f"添加数据库历史失败，回退到文件: {e}")
    items = load_history()
    items.append(entry)
    save_history(items)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    models = discover_models()
    history = load_history()
    selected_model = request.form.get('model_path') if request.method == 'POST' else None
    if request.method == 'POST':
        conf_str = request.form.get('conf', '0.6')
    else:
        conf_str = request.args.get('conf', '0.6')
    try:
        conf = float(conf_str)
    except Exception:
        conf = 0.6
    conf = max(0.0, min(1.0, conf))
    if not selected_model:
        if os.path.exists(DEFAULT_MODEL):
            selected_model = DEFAULT_MODEL
        elif models:
            selected_model = models[0]['path']
    if request.method == 'POST':
        if 'file' not in request.files:
            return '没有文件部分'
        file = request.files['file']
        if file.filename == '':
            return '未选择文件'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            try:
                with open(save_path, 'rb') as fh:
                    file_hash = hashlib.sha256(fh.read()).hexdigest()
            except Exception:
                file_hash = None

            predict_device = request.form.get('predict_device') or None
            use_cache = True
            try:
                use_cache = bool(request.form.get('use_cache', '1') in ('1', 'true', 'True', 'yes'))
            except Exception:
                use_cache = True
            cache_key = None
            if _redis_client and file_hash and use_cache:
                try:
                    key_raw = f"{selected_model}|{predict_device or 'auto'}|{file_hash}|conf={conf}|iou=0.5"
                    cache_key = 'pred:' + hashlib.sha1(key_raw.encode('utf-8')).hexdigest()
                except Exception:
                    cache_key = None
            if cache_key and _redis_client:
                try:
                    cached_name = _redis_client.get(cache_key)
                    if cached_name:
                        cached_path = os.path.join(app.config['UPLOAD_FOLDER'], cached_name)
                        if os.path.exists(cached_path):
                            filename = os.path.basename(cached_name)
                            result_img_path = cached_path
                            entry_labels = None
                            try:
                                prev = [h for h in load_history() if h.get('result_filename') == filename or h.get('filename') == filename]
                                if prev and isinstance(prev[-1].get('labels'), list):
                                    entry_labels = prev[-1].get('labels')
                            except Exception:
                                entry_labels = None
                            add_history_entry({
                                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'model_path': selected_model,
                                'model_label': _model_label(selected_model) if selected_model else '',
                                'filename': filename,
                                'result_filename': filename,
                                'labels': entry_labels,
                            })
                            history = load_history()
                            return render_template('index.html', models=models, selected_model=selected_model,
                                       filename=filename, result_img=filename, history=history, conf=conf)
                except Exception as e:
                    app.logger.debug(f"Redis 读取缓存失败: {e}")

            try:
                model = get_model_with_device(selected_model, device=predict_device)
            except Exception as e:
                return f'加载模型失败: {e}'
            if cache_key:
                cache_fname = f"cache_{hashlib.sha256(cache_key.encode('utf-8')).hexdigest()[:16]}.jpg"
                result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], cache_fname)
                result_basename = cache_fname
            else:
                result_basename = f'result_{filename}'
                result_img_path = os.path.join(app.config['UPLOAD_FOLDER'], result_basename)

            try:
                class_names = model.class_names or get_default_class_names()
                labels = model.predict_and_save(save_path, result_img_path, conf=conf, iou=0.5, class_names=class_names)
            except Exception as e:
                return f'识别失败: {e}'
            if cache_key and _redis_client:
                try:
                    _redis_client.set(cache_key, result_basename, ex=3600)
                except Exception as e:
                    app.logger.debug(f"Redis 写入缓存失败: {e}")
            entry_labels = labels if 'labels' in locals() and labels is not None else None
            if entry_labels is None:
                try:
                    prev = [h for h in load_history() if h.get('result_filename') == result_basename or h.get('filename') == result_basename]
                    if prev and isinstance(prev[-1].get('labels'), list):
                        entry_labels = prev[-1].get('labels')
                except Exception:
                    entry_labels = None
            add_history_entry({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': selected_model,
                'model_label': _model_label(selected_model) if selected_model else '',
                'filename': filename,
                'result_filename': f'result_{filename}',
                'labels': entry_labels,
            })
            history = load_history()
        return render_template('index.html', models=models, selected_model=selected_model,
                   filename=filename, result_img=f'result_{filename}', history=history, conf=conf)
    return render_template('index.html', models=models, selected_model=selected_model, history=history, conf=conf)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/history/clear', methods=['POST'])
def clear_history():
    if DB_ENABLED and SessionLocal is not None and UploadHistory is not None:
        try:
            with SessionLocal() as s:  
                s.query(UploadHistory).delete()  
                s.commit()
        except Exception as e:
            app.logger.error(f"清空数据库历史失败: {e}")
    else:
        save_history([])
    return redirect(url_for('upload_file'))
@app.route('/history/delete', methods=['POST'])
def delete_history_item():
    hid = request.form.get('id')
    fname = request.form.get('filename')
    if DB_ENABLED and SessionLocal is not None and UploadHistory is not None and hid:
        try:
            with SessionLocal() as s:
                obj = s.get(UploadHistory, int(hid))  
                if obj:
                    s.delete(obj)
                    s.commit()
        except Exception as e:
            app.logger.error(f"删除历史记录失败 (id={hid}): {e}")
        return redirect(url_for('upload_file'))
    if not fname and hid:
        fname = hid
    if fname:
        items = load_history()
        new_items = [it for it in items if it.get('filename') != fname and it.get('result_filename') != fname]
        save_history(new_items)
    return redirect(url_for('upload_file'))
def _gen_camera_frames(model):
    return _gen_camera_frames_with_device(model, 0)
def _gen_camera_frames_with_device(model, device, fps: float = 30.0, infer_every: int = 1, conf: float = 0.6, iou: float = 0.5, class_names: list[str] | None = None):
    if cv2 is None:
        raise RuntimeError('未安装 OpenCV。请先安装: pip install opencv-python')
    is_network = False
    try:
        if isinstance(device, str) and (device.startswith('rtsp://') or device.startswith('http://') or '://' in device):
            is_network = True
    except Exception:
        is_network = False
    def _open_cap(dev):
        try:
            c = cv2.VideoCapture(dev)
            return c
        except Exception as e:
            app.logger.debug(f"打开视频源失败 ({dev}): {e}")
            return None
    def _ffmpeg_frame_generator(url, fps=25):
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', url,
            '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            app.logger.debug(f"无法启动 ffmpeg 读取流: {e}")
            return
        width = None
        height = None
        try:
            p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', url], capture_output=True, text=True, timeout=5)
            if p.returncode == 0 and p.stdout:
                parts = p.stdout.strip().split(',')
                if len(parts) >= 2:
                    width = int(parts[0]); height = int(parts[1])
        except Exception:
            pass
        if width is None or height is None:
            width, height = 640, 480
        frame_size = width * height * 3
        try:
            while True:
                raw = proc.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                yield frame
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    cap = _open_cap(device)
    if cap is None or not cap.isOpened():
        if not is_network:
            raise RuntimeError(f'无法打开摄像头/流 ({device})')
    try:
        fps = float(fps)
    except Exception:
        fps = 30.0
    if fps <= 0:
        fps = 30.0
    fps = max(1.0, min(60.0, fps))
    try:
        infer_every = int(infer_every)
    except Exception:
        infer_every = 1
    infer_every = max(1, min(60, infer_every))
    delay = 1.0 / fps
    frame_count = 0
    try:
        backoff = 1.0
        max_backoff = 60.0
        while True:
            if cap is None or not cap.isOpened():
                if not is_network:
                    app.logger.error(f"本地摄像头无法打开或已断开: {device}")
                    break
                app.logger.info(f"尝试连接网络流 {device}（等待 {backoff}s）")
                ff_gen = None
                try:
                    ff_gen = _ffmpeg_frame_generator(device, fps=int(fps))
                except Exception:
                    ff_gen = None
                if ff_gen is not None:
                    app.logger.info(f"使用 ffmpeg 后备读取网络流: {device}")
                    try:
                        for frame in ff_gen:
                            if frame is None:
                                break
                            if frame_count % infer_every == 0:
                                try:
                                    dets = model.predict(frame, conf=conf, iou=iou)
                                    annotated = model.annotate(frame, dets, class_names=class_names)
                                except Exception as e:
                                    app.logger.debug(f"ffmpeg 识别异常，返回原始帧: {e}")
                                    annotated = frame
                            else:
                                annotated = frame
                            frame_count += 1
                            ret, buffer = cv2.imencode('.jpg', annotated)
                            if not ret:
                                continue
                            jpg = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
                            time.sleep(delay)
                    except Exception as e:
                        app.logger.debug(f"ffmpeg 读取流过程中出现错误: {e}")
                time.sleep(backoff)
                cap = _open_cap(device)
                if cap and cap.isOpened():
                    app.logger.info(f"已连接到网络流 {device}")
                    backoff = 1.0
                else:
                    backoff = min(max_backoff, backoff * 2)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                if not is_network:
                    app.logger.error(f"本地摄像头读取失败: {device}")
                    break
                continue
            if frame_count % infer_every == 0:
                try:
                    dets = model.predict(frame, conf=conf, iou=iou)
                    annotated = model.annotate(frame, dets, class_names=class_names)
                except Exception as e:
                    app.logger.debug(f"识别异常，返回原始帧: {e}")
                    annotated = frame
            else:
                annotated = frame
            frame_count += 1
            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret:
                continue
            jpg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            time.sleep(delay)
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
def parse_camera_device(s: str | None):
    if s is None or s == "":
        return 0
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return s
    return s
@app.route('/db')
def db_page():
    return render_template('db_visualize.html')
@app.route('/api/db_rows')
def api_db_rows():
    try:
        rows = []
        if DB_ENABLED and SessionLocal is not None and UploadHistory is not None:
            try:
                with SessionLocal() as s:
                    q = s.query(UploadHistory).order_by(UploadHistory.id.asc()).all()
                    rows = [r.to_dict() for r in q]
            except Exception as e:
                app.logger.debug(f"从数据库读取历史失败，回退到文件: {e}")
                rows = load_history()
        else:
            rows = load_history()
        return jsonify({'ok': True, 'rows': rows})
    except Exception as e:
        app.logger.error(f"api_db_rows 错误: {e}")
        return jsonify({'ok': False, 'error': str(e), 'rows': []}), 500
@app.route('/batch')
def batch_page():
    models = discover_models()
    return render_template('batch.html', models=models)
@app.route('/api/batch_infer', methods=['POST'])
def api_batch_infer():
    try:
        payload = request.get_json() or {}
        folder = payload.get('folder')
        model_path = payload.get('model')
        conf = float(payload.get('conf', 0.6))
        if not folder:
            return jsonify({'ok': False, 'error': '缺少 folder'})
        if not os.path.isabs(folder):
            folder = os.path.abspath(os.path.join(BASE_DIR, folder))
        if not os.path.isdir(folder):
            return jsonify({'ok': False, 'error': f'目录不存在: {folder}'})
        imgs = []
        for fn in os.listdir(folder):
            if not os.path.isfile(os.path.join(folder, fn)): continue
            if '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS:
                imgs.append(fn)
        imgs = sorted(imgs)
        counts = {}
        errors = []
        processed = 0
        for fn in imgs:
            in_path = os.path.join(folder, fn)
            out_name = f'batch_{int(time.time())}_{fn}'
            out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
            try:
                chosen = model_path or (DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else (discover_models()[0]['path'] if discover_models() else None))
                if not chosen:
                    errors.append({'file': fn, 'error': '未找到模型'})
                    continue
                model = get_model_with_device(chosen, device=None)
                class_names = model.class_names or get_default_class_names()
                labels = model.predict_and_save(in_path, out_path, conf=conf, iou=0.5, class_names=class_names)
                add_history_entry({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_path': chosen,
                    'model_label': _model_label(chosen) if chosen else '',
                    'filename': fn,
                    'result_filename': out_name,
                    'labels': labels,
                })
                processed += 1
                if labels:
                    for l in labels:
                        counts[l] = counts.get(l, 0) + 1
            except Exception as e:
                errors.append({'file': fn, 'error': str(e)})
        return jsonify({'ok': True, 'processed': processed, 'counts': counts, 'errors': errors})
    except Exception as e:
        app.logger.error(f'batch infer error: {e}')
        return jsonify({'ok': False, 'error': str(e)}), 500
def list_cameras(max_index: int = 6):
    indices = []
    if cv2 is None:
        return indices
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
        cap.release()
    return indices
def _cleanup_cache_files(retention_hours: int = 24):
    try:
        hours = int(retention_hours)
    except Exception:
        hours = 24
    cutoff = hours * 3600
    while True:
        try:
            now = time.time()
            for fname in os.listdir(app.config['UPLOAD_FOLDER']):
                if not fname.startswith('cache_'):
                    continue
                path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                try:
                    mtime = os.path.getmtime(path)
                    if now - mtime > cutoff:
                        os.remove(path)
                        app.logger.info(f"已删除过期缓存文件: {path}")
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(3600)  
@app.route('/api/onnx/providers')
def api_onnx_providers():
    try:
        import onnxruntime as _ort
        avail = _ort.get_available_providers() or []
        return jsonify({'ok': True, 'providers': avail})
    except Exception as e:
        app.logger.debug(f"获取 ONNX providers 失败: {e}")
        return jsonify({'ok': False, 'providers': []}), 200
try:
    cleanup_thread = threading.Thread(target=_cleanup_cache_files, args=(24,), daemon=True)
    cleanup_thread.start()
except Exception:
    pass
@app.route('/camera')
def camera_page():
    models = discover_models()
    selected = request.args.get('model_path')
    if not selected:
        if os.path.exists(DEFAULT_MODEL):
            selected = DEFAULT_MODEL
        elif models:
            selected = models[0]['path']
    device = request.args.get('device', '0')
    conf_str = request.args.get('conf', '0.6')
    try:
        conf = float(conf_str)
    except Exception:
        conf = 0.6
    conf = max(0.0, min(1.0, conf))
    available = list_cameras()
    return render_template('camera.html', models=models, selected_model=selected, device=device, available_cameras=available, conf=conf)
@app.route('/video_feed')
def video_feed():
    app.logger.info(f"video_feed 请求: args={dict(request.args)}")
    model_path = request.args.get('model_path')
    device_str = request.args.get('device', '0')
    predict_device = request.args.get('predict_device')
    device = parse_camera_device(device_str)
    if cv2 is None:
        return '未安装 OpenCV。请先安装: pip install opencv-python', 400
    fps = request.args.get('fps', '30')
    infer_every = request.args.get('infer_every', '1')
    conf_str = request.args.get('conf', '0.6')
    iou_str = request.args.get('iou', '0.5')
    if not model_path:
        if os.path.exists(DEFAULT_MODEL):
            model_path = DEFAULT_MODEL
        else:
            ms = discover_models()
            if not ms:
                return '未找到可用模型，请先放置 .onnx 权重到 models/ 下', 400
            model_path = ms[0]['path']
    try:
        model = get_model_with_device(model_path, device=predict_device)
    except Exception as e:
        return f'加载模型失败: {e}', 500
    try:
        conf = max(0.0, min(1.0, float(conf_str)))
    except Exception:
        conf = 0.6
    try:
        iou = max(0.0, min(1.0, float(iou_str)))
    except Exception:
        iou = 0.5
    try:
        class_names = model.class_names or get_default_class_names()
        return Response(_gen_camera_frames_with_device(model, device, fps=fps, infer_every=infer_every, conf=conf, iou=iou, class_names=class_names), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return f'启动视频流失败: {e}', 500
@app.route('/api/camera/diagnose', methods=['GET'])
def api_camera_diagnose():
    dev = request.args.get('device', '0')
    model_path = request.args.get('model_path')
    predict_device = request.args.get('predict_device')
    conf_str = request.args.get('conf', '0.6')
    try:
        conf = float(conf_str)
    except Exception:
        conf = 0.6
    try:
        d = parse_camera_device(dev)
    except Exception:
        d = dev
    if cv2 is None:
        return jsonify({'ok': False, 'error': 'OpenCV 未安装'}), 400
    cap = None
    try:
        cap = cv2.VideoCapture(d)
        opened = bool(cap and cap.isOpened())
        if not opened:
            return jsonify({'ok': False, 'error': '无法打开设备', 'device': str(d)}), 400
        ok, frame = cap.read()
        if not ok or frame is None:
            return jsonify({'ok': False, 'error': '读取帧失败', 'device': str(d)}), 500
        info = {'ok': True, 'device': str(d), 'frame_shape': list(frame.shape)}
        if model_path:
            try:
                model = get_model_with_device(model_path, device=predict_device)
                dets = model.predict(frame, conf=conf, iou=0.5)
                info['detections'] = len(dets)
                info['class_names'] = len(model.class_names) if model.class_names else None
            except Exception as e:
                info['model_error'] = str(e)
        return jsonify(info)
    except Exception as e:
        app.logger.error(f"camera diagnose 错误: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500
    finally:
        try:
            if cap:
                cap.release()
        except Exception:
            pass
@app.route('/logs', methods=['GET'])
def logs_page():
    defaults = {
        'tail': 400,
    }
    return render_template('logs.html', defaults=defaults)
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
