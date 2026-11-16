"""Validate the actually selected demo artifact through the ASGI application."""
import json
from pathlib import Path
from fastapi.testclient import TestClient
from telecom_churn.api import create_app

with TestClient(create_app(Path('runtime/model.pkl'))) as client:
    assert client.get('/health').status_code == 200
    meta = client.get('/model').json()
    body = json.loads(Path('docs/example-request.json').read_text())
    body['explain'] = meta['name'] == 'catboost'
    response = client.post('/predict', json=body)
    assert response.status_code == 200, response.text
    assert 0 <= response.json()['predictions'][0]['churn_probability'] <= 1
    assert client.post('/predict', json={'records': [{}]}).status_code == 422
print('Selected-artifact API smoke passed')
