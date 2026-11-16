"""Serve a trusted, explicitly selected artifact. No artifact uploads or arbitrary file paths."""
from contextlib import asynccontextmanager
import math
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from telecom_churn.pipeline import features_for, sha256


class PredictRequest(BaseModel):
    records: list[dict[str, float | str | None]] = Field(min_length=1, max_length=100)
    explain: bool = False


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        path = Path(model_path or os.getenv('CHURN_MODEL_PATH', '/app/runtime/model.pkl'))
        if not path.is_file():
            raise RuntimeError('Model missing. Train and select an artifact before starting the API.')
        # Trusted local artifact only; never deserialize user-supplied pickle files.
        with path.open('rb') as stream:
            app.state.artifact = pickle.load(stream)
        app.state.model_sha256 = sha256(path)
        yield

    app = FastAPI(title='Telecom Churn Prediction API', version='1.0.0', lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model': app.state.artifact['model_name']}

    @app.get('/model')
    def metadata():
        a = app.state.artifact
        return {'name': a['model_name'], 'threshold': a['threshold'],
                'features': a['manifest']['features'], 'categorical': a['manifest']['categorical'],
                'model_sha256': app.state.model_sha256, 'versions': a['versions']}

    @app.post('/predict')
    def predict(body: PredictRequest):
        a = app.state.artifact
        m = a['manifest']
        required = set(m['features'])
        categorical = set(m['categorical'])
        for i, row in enumerate(body.records):
            if set(row) != required:
                raise HTTPException(422, detail={'row': i, 'missing': sorted(required - set(row)),
                                                'unexpected': sorted(set(row) - required)})
            for key, val in row.items():
                if val is None:
                    continue
                if key in categorical:
                    if not isinstance(val, str):
                        raise HTTPException(422, f'Row {i}: {key} must be a string or null')
                elif isinstance(val, str) or not math.isfinite(val):
                    raise HTTPException(422, f'Row {i}: {key} must be a finite number or null')
        x = features_for(pd.DataFrame(body.records), m)
        prob = a['model'].predict_proba(x)[:, 1]
        predictions = [{'churn_probability': float(p), 'predicted_churn': int(p >= a['threshold'])} for p in prob]
        if body.explain:
            if a['model_name'] != 'catboost':
                raise HTTPException(422, 'Signed SHAP API explanations require a selected CatBoost model')
            from catboost import Pool
            values = a['model'].get_feature_importance(
                Pool(x, cat_features=m['categorical']), type='ShapValues', thread_count=2)
            for i, result in enumerate(predictions):
                ix = np.argsort(np.abs(values[i, :-1]))[-5:][::-1]
                result['base_log_odds'] = float(values[i, -1])
                result['top_factors'] = [{'feature': m['features'][j], 'contribution_log_odds': float(values[i, j])} for j in ix]
                result['other_contribution_log_odds'] = float(values[i, :-1].sum() - values[i, ix].sum())
        return {'model_sha256': app.state.model_sha256, 'threshold': a['threshold'], 'predictions': predictions}

    return app


app = create_app()
