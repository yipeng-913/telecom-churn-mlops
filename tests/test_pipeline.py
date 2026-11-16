import json
import numpy as np
import pandas as pd
import pytest
from telecom_churn.pipeline import prepare, read_raw, features_for, metrics, choose_threshold


def dataset(tmp_path):
    p=tmp_path/'data/raw';p.mkdir(parents=True)
    df=pd.DataFrame({'Customer_ID':range(200),'churn':[0,1]*100,
        'rev_Mean':[12.5,16.25]*100,'new_cell':['Y',None]*100})
    df.to_csv(p/'dataset.csv',sep=';',decimal=',',index=False)
    return df,p/'dataset.csv'


def test_source_format_and_disjoint_reproducible_splits(tmp_path):
    df,path=dataset(tmp_path)
    assert read_raw(path).rev_Mean.iloc[0]==12.5
    a=prepare(tmp_path,'run-a');b=prepare(tmp_path,'run-b')
    splits=[pd.read_pickle(f'{a}/{n}.pkl') for n in ['train','validation','test']]
    ids=[set(s.Customer_ID) for s in splits]
    assert len(set.union(*ids))==len(df)
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    assert [len(s) for s in splits]==[120,40,40]
    for n,s in zip(['train','validation','test'],splits):
        pd.testing.assert_frame_equal(s,pd.read_pickle(f'{b}/{n}.pkl'))


def test_duplicate_ids_fail_instead_of_leaking_across_splits(tmp_path):
    df,path=dataset(tmp_path);df.loc[1,'Customer_ID']=0
    df.to_csv(path,sep=';',decimal=',',index=False)
    with pytest.raises(ValueError,match='unique'):read_raw(path)


def test_missing_or_nonbinary_target_fails(tmp_path):
    df,path=dataset(tmp_path);df.loc[0,'churn']=2
    df.to_csv(path,sep=';',decimal=',',index=False)
    with pytest.raises(ValueError,match='binary'):read_raw(path)


def test_unlabeled_scoring_preserves_feature_contract_and_unknown_categories():
    m={'features':['rev_Mean','new_cell'],'categorical':['new_cell']}
    x=features_for(pd.DataFrame({'new_cell':['UNSEEN',None],'rev_Mean':[2,np.nan],'Customer_ID':[8,9]}),m)
    assert list(x)==m['features']
    assert x.new_cell.tolist()==['UNSEEN','__MISSING__']
    assert pd.isna(x.rev_Mean.iloc[1])
    with pytest.raises(ValueError,match='Missing scoring'):features_for(pd.DataFrame({'rev_Mean':[1]}),m)


def test_ranking_metrics_and_validation_threshold():
    y=np.array([0,0,1,1]);p=np.array([.1,.2,.8,.9])
    t=choose_threshold(y,p);m=metrics(y,p,t)
    assert m['roc_auc']==m['average_precision']==m['f1']==1
    assert m['confusion_matrix']==[[2,0],[0,2]]
    assert m['top20_lift']==2


def test_impossible_device_age_is_missing_for_training_and_scoring(tmp_path):
    df,path=dataset(tmp_path);df['eqpdays']=30;df.loc[0,'eqpdays']=-5
    df.to_csv(path,sep=';',decimal=',',index=False)
    prepared=prepare(tmp_path,'cleaning-test')
    m=json.loads(open(f'{prepared}/manifest.json').read())
    assert m['negative_device_ages_set_missing']==1
    splits=pd.concat([pd.read_pickle(f'{prepared}/{n}.pkl') for n in ['train','validation','test']])
    assert splits.loc[splits.Customer_ID==0,'eqpdays'].isna().all()
    assert features_for(df,m).loc[0:0,'eqpdays'].isna().all()
