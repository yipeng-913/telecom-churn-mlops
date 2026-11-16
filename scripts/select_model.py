"""Explicit local artifact selection. Evaluation results must exist before copying."""
import argparse
import json
from pathlib import Path
import shutil
from telecom_churn.pipeline import run_dir

p = argparse.ArgumentParser()
p.add_argument('--root', default='.')
p.add_argument('--run-id', required=True)
a = p.parse_args()
root = Path(a.root)
run = run_dir(root, a.run_id).name
metrics = json.loads((root / 'reports' / run / 'metrics.json').read_text())
source = root / 'models' / run / 'model.pkl'
from telecom_churn.pipeline import sha256
if metrics['model_sha256'] != sha256(source):
    raise ValueError('Evaluated artifact hash does not match model')
runtime = root / 'runtime'
runtime.mkdir(exist_ok=True)
# This selected deployment copy must be readable by the non-root container UID.
# Leave the original training artifact and private source directories unchanged.
runtime.chmod(0o755)
shutil.copyfile(source, runtime / 'model.pkl')
(runtime / 'model.pkl').chmod(0o644)
print('Selected evaluated artifact:', run, 'AP:', metrics['average_precision'])
