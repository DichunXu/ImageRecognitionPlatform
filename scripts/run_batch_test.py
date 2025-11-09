import os
import json
from app import app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(BASE_DIR)
UPLOADS = os.path.join(CODE_DIR, 'uploads')
os.makedirs(UPLOADS, exist_ok=True)
print('Running batch inference test: folder=uploads, model=None (default), conf=0.6')
with app.test_client() as c:
    payload = {'folder': 'uploads', 'model': None, 'conf': 0.6}
    resp = c.post('/api/batch_infer', json=payload)
    print('HTTP', resp.status_code)
    try:
        j = resp.get_json()
        print(json.dumps(j, ensure_ascii=False, indent=2))
    except Exception:
        print(resp.get_data(as_text=True))
