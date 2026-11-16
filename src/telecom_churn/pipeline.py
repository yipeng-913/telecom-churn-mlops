"""Three independently runnable stages. All cross-task state is in run-scoped files."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import pickle
import platform
from pathlib import Path
import re
from importlib.metadata import version

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, log_loss,
    brier_score_loss, precision_recall_curve, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
LOG = logging.getLogger(__name__)
TARGET, ID = 'churn', 'Customer_ID'

def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def run_dir(root, run_id):
    # Airflow run ids include punctuation; suffix prevents collisions after sanitizing.
    token = re.sub(r'[^a-zA-Z0-9_-]', '_', run_id)[:90]
    token += '_' + hashlib.sha256(run_id.encode()).hexdigest()[:10]
    p = Path(root) / 'data' / 'processed' / token
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_raw(path):
    df = pd.read_csv(path, sep=';', decimal=',', na_values=['', 'NA', 'NULL'])
    required = {TARGET, ID}
    if not required.issubset(df):
        raise ValueError(f'Missing columns: {required - set(df)}. Expected semicolon-delimited CSV.')
    if df.empty or df[ID].isna().any() or df[ID].duplicated().any():
        raise ValueError('Expected nonempty data and unique, nonmissing customer IDs')
    if df[TARGET].isna().any() or set(df[TARGET].unique()) != {0, 1}:
        raise ValueError('churn must contain both binary labels 0 and 1 without missing values')
    numeric = df.select_dtypes(include='number').columns
    if np.isinf(df[numeric].to_numpy(dtype=float)).any():
        raise ValueError('Infinite numeric values are not supported')
    return df

def prepare(root, run_id):
    root = Path(root)
    source = root / 'data/raw/dataset.csv'
    df = read_raw(source)
    invalid_device_ages = int((df['eqpdays'] < 0).sum()) if 'eqpdays' in df else 0
    if 'eqpdays' in df:
        df.loc[df['eqpdays'] < 0, 'eqpdays'] = np.nan
    train, rest = train_test_split(df, test_size=.4, stratify=df[TARGET], random_state=SEED)
    valid, test = train_test_split(rest, test_size=.5, stratify=rest[TARGET], random_state=SEED)
    path = run_dir(root, run_id)
    for name, split in [('train',train), ('validation',valid), ('test',test)]:
        split.to_pickle(path / f'{name}.pkl')
    features = [c for c in df.columns if c not in (TARGET, ID)]
    categorical = df[features].select_dtypes(exclude='number').columns.tolist()
    manifest = dict(run_id=run_id, source_sha256=sha256(source), seed=SEED,
        rows=len(df), columns=len(df.columns), features=features, categorical=categorical,
        negative_device_ages_set_missing=invalid_device_ages,
        split_rows={k:len(s) for k,s in [('train',train),('validation',valid),('test',test)]},
        split_prevalence={k:float(s[TARGET].mean()) for k,s in [('train',train),('validation',valid),('test',test)]},
        split_sha256={k:sha256(path/f'{k}.pkl') for k in ['train','validation','test']},
        missing_fraction={c:float(v) for c,v in df.isna().mean().items()},
        constant_columns=df.columns[df.nunique(dropna=False)<=1].tolist())
    dump_json(path/'manifest.json',manifest)
    LOG.info('Prepared data: %s', manifest['split_rows'])
    return str(path)

def features_for(df, manifest):
    missing = set(manifest['features']) - set(df)
    if missing:
        raise ValueError(f'Missing scoring features: {sorted(missing)}')
    x = df[manifest['features']].copy()
    if 'eqpdays' in x:
        x.loc[x['eqpdays'] < 0, 'eqpdays'] = np.nan
    for c in manifest['categorical']:
        x[c] = x[c].fillna('__MISSING__').astype(str)
    return x

def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y,p)
    scores = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(thresholds[np.argmax(scores)])

def metrics(y,p,threshold):
    pred = np.asarray(p) >= threshold
    k = max(1, int(np.ceil(.2*len(y))))
    top = np.argsort(-np.asarray(p),kind='stable')[:k]
    y = np.asarray(y)
    prevalence = float(y.mean())
    precision_top = float(y[top].mean())
    return dict(rows=len(y), prevalence=prevalence, average_precision=float(average_precision_score(y,p)),
        roc_auc=float(roc_auc_score(y,p)), log_loss=float(log_loss(y,p)),
        brier=float(brier_score_loss(y,p)), threshold=float(threshold),
        precision=float(precision_score(y,pred,zero_division=0)), recall=float(recall_score(y,pred,zero_division=0)),
        f1=float(f1_score(y,pred,zero_division=0)), confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
        top20_count=k, top20_precision=precision_top, top20_lift=precision_top/prevalence,
        top20_capture=float(y[top].sum()/y.sum()))

def train(root, prepared):
    root, path = Path(root), Path(prepared)
    m = json.loads((path/'manifest.json').read_text())
    tr, va = pd.read_pickle(path/'train.pkl'), pd.read_pickle(path/'validation.pkl')
    xt, xv = features_for(tr,m), features_for(va,m)
    numeric = [c for c in m['features'] if c not in m['categorical']]
    preprocess = ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
            ('scale',StandardScaler())]),numeric),
        ('categorical',OneHotEncoder(handle_unknown='ignore'),m['categorical'])])
    baseline = Pipeline([('preprocess',preprocess),('model',LogisticRegression(max_iter=1500,solver='liblinear',random_state=SEED))])
    baseline.fit(xt,tr[TARGET])
    model = CatBoostClassifier(iterations=1200, depth=6, learning_rate=.05, loss_function='Logloss',
        eval_metric='AUC', random_seed=SEED, thread_count=4, l2_leaf_reg=5,
        allow_writing_files=False, verbose=False)
    model.fit(xt,tr[TARGET],cat_features=m['categorical'],eval_set=(xv,va[TARGET]),
        early_stopping_rounds=100,verbose=False)
    candidates={'logistic':baseline,'catboost':model}
    scores = {}
    for name,candidate in candidates.items():
        p = candidate.predict_proba(xv)[:,1]
        threshold = choose_threshold(va[TARGET],p)
        scores[name] = metrics(va[TARGET],p,threshold)
    winner = max(scores,key=lambda k:scores[k]['average_precision'])
    model_dir = root/'models'/path.name
    model_dir.mkdir(parents=True,exist_ok=True)
    artifact = dict(model=candidates[winner], explainer_model=model, model_name=winner,
        threshold=scores[winner]['threshold'], manifest=m, validation_metrics=scores,
        catboost_best_iteration=int(model.get_best_iteration()),
        versions={p:version(p) for p in ['pandas','numpy','scikit-learn','catboost']},
        python=platform.python_version())
    target = model_dir/'model.pkl'
    temporary = model_dir/'model.pkl.tmp'
    with temporary.open('wb') as f: pickle.dump(artifact,f,pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)
    dump_json(model_dir/'metadata.json',{k:v for k,v in artifact.items() if k not in ['model','explainer_model']})
    LOG.info('Selected %s using validation AP; scores=%s',winner,scores)
    return str(target)

def plot_setup():
    os.environ.setdefault('MPLCONFIGDIR','/tmp/churn-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False})
    return plt

def bootstrap_auc_ap(y,p,n=200):
    rng=np.random.default_rng(SEED)
    rows=[]
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))==2:
            rows.append([roc_auc_score(y[ix],p[ix]),average_precision_score(y[ix],p[ix])])
    a=np.asarray(rows)
    return {k:[float(v) for v in np.quantile(a[:,i],[.025,.975])] for i,k in enumerate(['roc_auc','average_precision'])}

def evaluate(root, prepared, model_path):
    root,path=Path(root),Path(prepared)
    # Only load trusted artifacts generated by this project.
    with open(model_path,'rb') as f: artifact=pickle.load(f)
    m=artifact['manifest']
    on_disk=json.loads((path/'manifest.json').read_text())
    if m['run_id']!=on_disk['run_id'] or m['source_sha256']!=on_disk['source_sha256']:
        raise ValueError('Model and data manifest do not match')
    for name in ['train','validation','test']:
        if sha256(path/f'{name}.pkl')!=m['split_sha256'][name]:
            raise ValueError(f'{name} split changed after preparation')
    te=pd.read_pickle(path/'test.pkl')
    x=features_for(te,m); y=te[TARGET].to_numpy()
    p=artifact['model'].predict_proba(x)[:,1]
    result=metrics(y,p,artifact['threshold'])
    result.update(model_name=artifact['model_name'],source_sha256=m['source_sha256'],
        model_sha256=sha256(model_path),bootstrap_95_percentile_ci=bootstrap_auc_ap(y,p),
        validation=artifact['validation_metrics'],catboost_best_iteration=artifact['catboost_best_iteration'])
    report=root/'reports'/path.name
    (report/'figures').mkdir(parents=True,exist_ok=True)
    dump_json(report/'metrics.json',result)
    scored=pd.DataFrame({ID:te[ID].to_numpy(),'actual_churn':y,'churn_probability':p,
        'predicted_churn':(p>=artifact['threshold']).astype(int)})
    scored.to_csv(report/'holdout_predictions.csv',index=False)
    plt=plot_setup()
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    fpr,tpr,_=roc_curve(y,p); ax[0].plot(fpr,tpr,label=f"AUC={result['roc_auc']:.3f}")
    ax[0].plot([0,1],[0,1],'--',color='gray');ax[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Hold-out ROC');ax[0].legend()
    pr,rc,_=precision_recall_curve(y,p);ax[1].plot(rc,pr,label=f"AP={result['average_precision']:.3f}")
    ax[1].axhline(y.mean(),linestyle='--',color='gray');ax[1].set(xlabel='Recall',ylabel='Precision',title='Hold-out precision / recall');ax[1].legend()
    from sklearn.calibration import calibration_curve
    frac,mean=calibration_curve(y,p,n_bins=10,strategy='quantile')
    ax[2].plot(mean,frac,'o-');ax[2].plot([0,1],[0,1],'--',color='gray');ax[2].set(xlabel='Mean predicted risk',ylabel='Observed churn',title='Calibration (sample population)')
    fig.tight_layout();fig.savefig(report/'figures/evaluation.png');plt.close(fig)
    # SHAP is built into CatBoost. It explains raw log odds, not percentage points.
    if artifact['model_name']=='catboost':
        sample=x.sample(n=min(1000,len(x)),random_state=SEED)
        values=artifact['model'].get_feature_importance(Pool(sample,cat_features=m['categorical']),type='ShapValues',thread_count=4)
        raw=artifact['model'].predict(sample,prediction_type='RawFormulaVal')
        if not np.allclose(values.sum(axis=1),raw,atol=1e-6):
            raise AssertionError('SHAP additivity check failed')
        importance=pd.DataFrame({'feature':m['features'],'mean_abs_shap_log_odds':np.abs(values[:,:-1]).mean(axis=0)}).sort_values('mean_abs_shap_log_odds',ascending=False)
        importance.to_csv(report/'feature_importance.csv',index=False)
        top=importance.head(20).iloc[::-1]
        fig,ax=plt.subplots(figsize=(9,7));ax.barh(top.feature,top.mean_abs_shap_log_odds,color='#167d9a');ax.set(xlabel='Mean |SHAP| (log odds)',title=f'Global impact on hold-out sample (n={len(sample)})');fig.tight_layout();fig.savefig(report/'figures/global_shap.png');plt.close(fig)
        # Stable sample includes high, middle, and low risk customers; no target-based selection.
        ranks=np.argsort(p,kind='stable');chosen=[ranks[0],ranks[len(p)//2],*ranks[-3:]]
        local=x.iloc[chosen]
        sv=artifact['model'].get_feature_importance(Pool(local,cat_features=m['categorical']),type='ShapValues',thread_count=4)
        records=[]
        for row_idx,position in enumerate(chosen):
            for j,feature in enumerate(m['features']):
                records.append({ID:int(te.iloc[position][ID]),'churn_probability':float(p[position]),'feature':feature,
                    'value':str(local.iloc[row_idx][feature]),'shap_log_odds':float(sv[row_idx,j]),'base_log_odds':float(sv[row_idx,-1])})
        pd.DataFrame(records).to_csv(report/'individual_explanations.csv',index=False)
        last=sv[-1,:-1];ix=np.argsort(np.abs(last))[-12:]
        fig,ax=plt.subplots(figsize=(9,5));ax.barh(np.asarray(m['features'])[ix],last[ix],color=np.where(last[ix]>0,'#d55a49','#167d9a'))
        ax.axvline(0,color='gray');ax.set(xlabel='SHAP contribution to log odds (+ increases risk)',title=f"Customer {int(te.iloc[chosen[-1]][ID])}: risk {p[chosen[-1]]:.1%}");fig.tight_layout();fig.savefig(report/'figures/local_shap.png');plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for ax,feature in zip(axes.flat,importance.head(6).feature):
            j=m['features'].index(feature)
            if feature in m['categorical']:
                tmp=pd.DataFrame({'value':sample[feature].to_numpy(),'shap':values[:,j]}).groupby('value').shap.mean().sort_values()
                tmp.plot.bar(ax=ax,color='#167d9a');ax.tick_params(axis='x',labelsize=7)
            else:
                ax.scatter(sample[feature],values[:,j],s=5,alpha=.35,color='#167d9a')
            ax.axhline(0,color='gray',lw=.6);ax.set(title=feature,ylabel='SHAP (log odds)')
        fig.tight_layout();fig.savefig(report/'figures/shap_dependence.png');plt.close(fig)
    else:
        # Linear coefficient contributions exactly explain selected logistic model raw log odds.
        transformed=artifact['model']['preprocess'].transform(x)
        names=artifact['model']['preprocess'].get_feature_names_out()
        coef=artifact['model']['model'].coef_[0]
        pd.DataFrame({'transformed_feature':names,'coefficient':coef}).to_csv(report/'linear_coefficients.csv',index=False)
        from sklearn.inspection import permutation_importance
        imp=permutation_importance(artifact['model'],x,y,scoring='average_precision',n_repeats=3,random_state=SEED,n_jobs=1)
        pd.DataFrame({'feature':m['features'],'ap_decrease':imp.importances_mean}).sort_values('ap_decrease',ascending=False).to_csv(report/'feature_importance.csv',index=False)
        selected=np.argsort(p)[-5:]; transformed=transformed[selected].toarray() if hasattr(transformed,'toarray') else transformed[selected]
        rows=[]
        for i,pos in enumerate(selected):
            for j,name in enumerate(names):
                rows.append({ID:int(te.iloc[pos][ID]),'feature':name,'shap_log_odds':float(transformed[i,j]*coef[j]),'base_log_odds':float(artifact['model']['model'].intercept_[0])})
        pd.DataFrame(rows).to_csv(report/'individual_explanations.csv',index=False)
    # Descriptive tables below intentionally use training data only.
    tr=pd.read_pickle(path/'train.pkl')
    tr.isna().mean().sort_values(ascending=False).rename('missing_fraction').to_csv(report/'missingness.csv')
    tables=[]
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,c in zip(axes.flat,['eqpdays','months','mou_Mean','change_mou','custcare_Mean','rev_Mean']):
        bins=pd.qcut(tr[c],q=5,duplicates='drop')
        tmp=tr.groupby(bins,observed=True)[TARGET].agg(['size','mean']).reset_index()
        tmp.columns=['bin','customers','churn_rate'];tmp['feature']=c
        tmp['standard_error']=np.sqrt(tmp.churn_rate*(1-tmp.churn_rate)/tmp.customers)
        tmp['bin']=tmp['bin'].map(lambda v: f'{v.left:.1f}–{v.right:.1f}').astype(str);tables.append(tmp)
        ax.bar(range(len(tmp)),tmp.churn_rate,color='#167d9a',yerr=1.96*tmp.standard_error,capsize=3)
        ax.set_xticks(range(len(tmp)),tmp['bin'],rotation=35,ha='right',fontsize=7)
        ax.set(title=c,ylabel='Observed churn rate');ax.set_ylim(0,1)
    pd.concat(tables,ignore_index=True).to_csv(report/'segment_churn.csv',index=False)
    fig.suptitle('Training-set associations (95% normal-approximation error bars)');fig.tight_layout();fig.savefig(report/'figures/eda_segments.png');plt.close(fig)
    # All variables receive a univariate association summary.
    all_tables=[]
    for c in m['features']:
        if c in m['categorical']:
            bins=tr[c].fillna('__MISSING__')
        else:
            bins=pd.qcut(tr[c],q=5,duplicates='drop').astype('string').fillna('__MISSING__')
        tmp=tr.groupby(bins,observed=True)[TARGET].agg(['size','mean']).reset_index()
        tmp.columns=['bin','customers','churn_rate'];tmp['feature']=c;tmp['bin']=tmp['bin'].astype(str);all_tables.append(tmp)
    pd.concat(all_tables,ignore_index=True).to_csv(report/'all_feature_segments.csv',index=False)
    LOG.info('HOLD-OUT METRICS: %s',json.dumps(result))
    return str(report)

def score(model_path, csv_path, output):
    with open(model_path,'rb') as f: artifact=pickle.load(f)
    df=pd.read_csv(csv_path,sep=';',decimal=',')
    x=features_for(df,artifact['manifest'])
    p=artifact['model'].predict_proba(x)[:,1]
    result=pd.DataFrame({'churn_probability':p,'predicted_churn':(p>=artifact['threshold']).astype(int)})
    if ID in df: result.insert(0,ID,df[ID].to_numpy())
    result.to_csv(output,index=False)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','train','evaluate','run','score'])
    parser.add_argument('--root',default=os.getenv('CHURN_PROJECT_ROOT','/opt/airflow'))
    parser.add_argument('--run-id',default='local-demo')
    parser.add_argument('--prepared');parser.add_argument('--model');parser.add_argument('--input');parser.add_argument('--output')
    a=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if a.stage=='score':
        if not all([a.model,a.input,a.output]):parser.error('score requires --model --input --output')
        score(a.model,a.input,a.output);return
    prepared=a.prepared or str(run_dir(a.root,a.run_id))
    if a.stage in ['prepare','run']:prepared=prepare(a.root,a.run_id)
    model=a.model
    if a.stage in ['train','run']:model=train(a.root,prepared)
    if a.stage in ['evaluate','run']:
        if not model:parser.error('evaluate requires --model')
        print(evaluate(a.root,prepared,model))

if __name__=='__main__':main()
