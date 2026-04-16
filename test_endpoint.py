import requests

url = 'http://127.0.0.1:5000/train_and_candidates'
filepath = 'frontend/sample_candidates.csv'

try:
    with open(filepath, 'rb') as f:
        files = {'csv': f}
        data = {
            'job_desc': 'Looking for a Python developer with SQL experience',
            'num_candidates': '5'
        }
        r = requests.post(url, files=files, data=data, timeout=10)
        print('Status Code:', r.status_code)
        print('Response:', r.text)
except Exception as e:
    print('Error:', str(e))
