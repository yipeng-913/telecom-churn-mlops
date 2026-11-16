"""Generate independent synthetic telecom data; no assessment records or fitted source models."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate(root, rows=6000, seed=2026):
    if rows < 500:
        raise ValueError('Use at least 500 customers for stratified evaluation')
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 1000, rows)
    tenure = rng.integers(6, 61, rows)
    usage = np.clip(rng.gamma(2, 200, rows), 0, 3000)
    change = rng.normal(0, 180, rows)
    care = rng.poisson(1.3, rows).astype(float)
    failed = rng.poisson(4, rows).astype(float)
    charge = rng.uniform(15, 130, rows)
    area = rng.choice(['North', 'South', 'East', 'West'], rows)
    # Explicit synthetic, non-linear signal; not estimated from confidential/customer data.
    logit = (-2.0 + 1.5 * (age > 360) + 1.4 * (change < -100)
             + 1.0 * ((tenure > 11) & (tenure < 18))
             + 1.2 * ((charge > 80) & (usage < 200))
             + 0.07 * failed + 0.16 * care - 0.0006 * usage)
    risk = 1 / (1 + np.exp(-logit))
    df = pd.DataFrame({
        'Customer_ID': np.arange(1, rows + 1), 'churn': rng.binomial(1, risk),
        'eqpdays': age, 'months': tenure, 'mou_Mean': usage, 'change_mou': change,
        'custcare_Mean': care, 'rev_Mean': charge + rng.normal(0, 5, rows),
        'totmrc_Mean': charge, 'drop_vce_Mean': failed,
        'uniqsubs': rng.integers(1, 5, rows), 'hnd_price': rng.choice([99., 199., 399.], rows),
        'area': area, 'dualband': rng.choice(['Y', 'N'], rows),
    })
    for c in ['eqpdays', 'mou_Mean', 'rev_Mean', 'area']:
        df.loc[rng.random(rows) < .03, c] = np.nan
    path = Path(root) / 'data/raw/dataset.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'{path} exists; choose a new --root to preserve the source')
    df.to_csv(path, sep=';', decimal=',', index=False)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='.')
    p.add_argument('--rows', type=int, default=6000)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    print(generate(a.root, a.rows, a.seed))


if __name__ == '__main__':
    main()
