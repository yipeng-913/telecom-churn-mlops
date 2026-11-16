import pickle
import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient
from telecom_churn.api import create_app


@pytest.fixture()
def client(tmp_path):
    x = pd.DataFrame({'eqpdays': [20., 400., 90., 600.] * 8,
                      'area': ['North', 'South', 'North', 'South'] * 8})
    y = [0, 1, 0, 1] * 8
    model = CatBoostClassifier(iterations=10, depth=2, verbose=False, allow_writing_files=False, thread_count=1)
    model.fit(x, y, cat_features=['area'])
    artifact = {'model': model, 'model_name': 'catboost', 'threshold': .5,
                'manifest': {'features': ['eqpdays', 'area'], 'categorical': ['area']}, 'versions': {}}
    path = tmp_path / 'model.pkl'
    with path.open('wb') as f:
        pickle.dump(artifact, f)
    with TestClient(create_app(path)) as c:
        yield c


def test_health_schema_and_unknown_category(client):
    assert client.get('/health').json()['status'] == 'ok'
    assert client.get('/model').json()['features'] == ['eqpdays', 'area']
    response = client.post('/predict', json={'records': [{'eqpdays': 390, 'area': 'UNSEEN'}]})
    assert response.status_code == 200
    assert 0 <= response.json()['predictions'][0]['churn_probability'] <= 1


def test_explanation_reconstructs_probability(client):
    response = client.post('/predict', json={'records': [{'eqpdays': None, 'area': 'North'}], 'explain': True})
    assert response.status_code == 200
    row = response.json()['predictions'][0]
    raw = row['base_log_odds'] + sum(f['contribution_log_odds'] for f in row['top_factors']) + row['other_contribution_log_odds']
    assert np.isclose(1 / (1 + np.exp(-raw)), row['churn_probability'])


@pytest.mark.parametrize('record', [
    {'eqpdays': 2}, {'eqpdays': 2, 'area': 'North', 'churn': 1},
    {'eqpdays': 'bad', 'area': 'North'}, {'eqpdays': 2, 'area': 42},
])
def test_invalid_feature_contract_is_rejected(client, record):
    assert client.post('/predict', json={'records': [record]}).status_code == 422


def test_batch_bounds(client):
    assert client.post('/predict', json={'records': []}).status_code == 422
    rows = [{'eqpdays': 10, 'area': 'North'}] * 101
    assert client.post('/predict', json={'records': rows}).status_code == 422


def test_missing_model_fails_startup(tmp_path):
    with pytest.raises(RuntimeError, match='Model missing'):
        with TestClient(create_app(tmp_path / 'missing.pkl')):
            pass
