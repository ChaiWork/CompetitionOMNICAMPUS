"""
================================================================================
GCI Global 2026 Home Credit Default Risk Competition
Master Solution Pipeline (Reproducible 5-Fold Gradient Boosting Ensemble)
Author: Senior Data Scientist & ML Competition Specialist
Evaluation Metric: ROC AUC
================================================================================
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. Configuration & Reproducibility
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
N_SPLITS = 5
INPUT_DIR = Path('input')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------
# 2. Domain Feature Engineering
# ------------------------------------------------------------------------------
def build_features(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Constructs domain-informed features based on credit risk modeling principles.
    Treats anomalies, computes financial leverage ratios, aggregates external credit scores,
    and constructs interaction frequency encodings.
    """
    n_train = len(df_train)
    df = pd.concat([df_train.drop('TARGET', axis=1, errors='ignore'), df_test], axis=0).reset_index(drop=True)
    
    # --- A. Data Cleaning & Anomaly Correction ---
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'F')
    df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].replace('XNA', np.nan)
    df.loc[df['OWN_CAR_AGE'] >= 60, 'OWN_CAR_AGE'] = np.nan
    
    # --- B. Financial & Leverage Ratios ---
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1e-5)
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1e-5)
    df['INCOME_CREDIT_DIFF'] = df['AMT_INCOME_TOTAL'] - df['AMT_CREDIT']
    df['GOODS_CREDIT_DIFF'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1e-5)
    df['ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_GOODS_PRICE'] + 1e-5)
    
    # --- C. External Credit Source Aggregations ---
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCES_STD'] = df[ext_cols].std(axis=1).fillna(0)
    df['EXT_SOURCES_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCES_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_SUM'] = df['EXT_SOURCE_1'] + df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']
    df['EXT_SOURCES_NA_COUNT'] = df[ext_cols].isna().sum(axis=1)
    
    # --- D. Age & Duration Dynamics ---
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
    df['EXT_SOURCE_1_OVER_AGE'] = df['EXT_SOURCE_1'] / (df['AGE_YEARS'] + 1e-5)
    df['EXT_SOURCE_2_OVER_AGE'] = df['EXT_SOURCE_2'] / (df['AGE_YEARS'] + 1e-5)
    df['EXT_SOURCE_3_OVER_AGE'] = df['EXT_SOURCE_3'] / (df['AGE_YEARS'] + 1e-5)
    df['EXT_SOURCE_1_EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_2_EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1e-5)
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] + 1e-5)
    df['PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED'] + 1e-5)
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / (df['DAYS_BIRTH'] + 1e-5)
    df['REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / (df['DAYS_BIRTH'] + 1e-5)
    df['CAR_TO_AGE_RATIO'] = df['OWN_CAR_AGE'] / (df['AGE_YEARS'] + 1e-5)
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / (df['EMPLOYED_YEARS'] + 1e-5)
    
    # --- E. Bureau Enquiries & Contact Details ---
    bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON', 
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    df['BUREAU_REQ_TOTAL'] = df[bureau_cols].sum(axis=1)
    df['BUREAU_REQ_RATIO_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / (df['AMT_REQ_CREDIT_BUREAU_YEAR'] + 1e-5)
    df['CONTACT_INFO_TOTAL'] = df['FLAG_MOBIL'] + df['FLAG_EMP_PHONE'] + df['FLAG_WORK_PHONE']
    
    # --- F. Joint Frequency Interaction Encodings ---
    cat_pairs = [
        ('OCCUPATION_TYPE', 'ORGANIZATION_TYPE'),
        ('NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE'),
        ('CODE_GENDER', 'NAME_EDUCATION_TYPE'),
        ('NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE')
    ]
    for c1, c2 in cat_pairs:
        pair_col = f'{c1}___{c2}'
        df[pair_col] = df[c1].astype(str) + '_' + df[c2].astype(str)
        freq = df[pair_col].value_counts()
        df[f'{pair_col}_FREQ'] = df[pair_col].map(freq)
        df.drop(pair_col, axis=1, inplace=True)
        
    for c in ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']:
        freq = df[c].value_counts()
        df[f'{c}_FREQ'] = df[c].map(freq)
        
    # --- G. Categorical Encoding ---
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    X_train = df.iloc[:n_train].copy().reset_index(drop=True)
    X_test = df.iloc[n_train:].copy().reset_index(drop=True)
    
    return X_train, X_test, cat_cols


# ------------------------------------------------------------------------------
# 3. Main Pipeline Execution
# ------------------------------------------------------------------------------
def run_pipeline():
    start_time = time.time()
    print('Loading datasets...')
    train = pd.read_csv(INPUT_DIR / 'train.csv')
    test = pd.read_csv(INPUT_DIR / 'test.csv')
    sample_sub = pd.read_csv(INPUT_DIR / 'sample_submission.csv')
    y = train['TARGET'].values
    
    print(f'Train shape: {train.shape}, Test shape: {test.shape}')
    
    # Feature construction
    X_train, X_test, cat_cols = build_features(train, test)
    features = [c for c in X_train.columns if c not in ['TARGET', 'SK_ID_CURR']]
    print(f'Constructed {len(features)} total predictive features.')
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    oof_lgb = np.zeros(len(train))
    test_lgb = np.zeros(len(test))
    oof_xgb = np.zeros(len(train))
    test_xgb = np.zeros(len(test))
    oof_cb  = np.zeros(len(train))
    test_cb  = np.zeros(len(test))
    
    # --- 1. LightGBM ---
    print('\n[1/3] Training 5-Fold LightGBM...')
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.04,
        'num_leaves': 31,
        'max_depth': 6,
        'colsample_bytree': 0.7,
        'subsample': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.loc[trn_idx, features], y[trn_idx]
        X_va, y_va = X_train.loc[val_idx, features], y[val_idx]
        X_te = X_test[features]
        
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        
        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=1500,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        oof_lgb[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        test_lgb += model.predict(X_te, num_iteration=model.best_iteration) / N_SPLITS
    print(f'-> LightGBM 5-Fold OOF AUC: {roc_auc_score(y, oof_lgb):.6f}')
    
    # --- 2. XGBoost ---
    print('\n[2/3] Training 5-Fold XGBoost...')
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.04,
        'max_depth': 5,
        'colsample_bytree': 0.7,
        'subsample': 0.8,
        'alpha': 1.5,
        'lambda': 5.0,
        'tree_method': 'hist',
        'enable_categorical': True,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.loc[trn_idx, features], y[trn_idx]
        X_va, y_va = X_train.loc[val_idx, features], y[val_idx]
        X_te = X_test[features]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval   = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dtest  = xgb.DMatrix(X_te, enable_categorical=True)
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        oof_xgb[val_idx] = model.predict(dval)
        test_xgb += model.predict(dtest) / N_SPLITS
    print(f'-> XGBoost 5-Fold OOF AUC: {roc_auc_score(y, oof_xgb):.6f}')
    
    # --- 3. CatBoost ---
    print('\n[3/3] Training 5-Fold CatBoost...')
    X_train_cb = X_train[features].copy()
    X_test_cb  = X_test[features].copy()
    for c in cat_cols:
        X_train_cb[c] = X_train_cb[c].astype('object').fillna('MISSING').astype(str)
        X_test_cb[c]  = X_test_cb[c].astype('object').fillna('MISSING').astype(str)
        
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train_cb, y)):
        X_tr, y_tr = X_train_cb.iloc[trn_idx], y[trn_idx]
        X_va, y_va = X_train_cb.iloc[val_idx], y[val_idx]
        
        train_pool = cb.Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool   = cb.Pool(X_va, y_va, cat_features=cat_cols)
        test_pool  = cb.Pool(X_test_cb, cat_features=cat_cols)
        
        cb_model = cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.04,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=RANDOM_SEED,
            task_type='CPU',
            thread_count=-1,
            verbose=False
        )
        cb_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
        oof_cb[val_idx] = cb_model.predict_proba(val_pool)[:, 1]
        test_cb += cb_model.predict_proba(test_pool)[:, 1] / N_SPLITS
    print(f'-> CatBoost 5-Fold OOF AUC: {roc_auc_score(y, oof_cb):.6f}')
    
    # --- Ensembling & Blending ---
    print('\n[Ensemble] Optimizing Blending Weights...')
    rank_lgb = rankdata(oof_lgb) / len(oof_lgb)
    rank_xgb = rankdata(oof_xgb) / len(oof_xgb)
    rank_cb  = rankdata(oof_cb) / len(oof_cb)
    
    def loss_func(weights):
        w1, w2, w3 = weights
        pred = w1 * rank_lgb + w2 * rank_xgb + w3 * rank_cb
        return -roc_auc_score(y, pred)
        
    res = minimize(loss_func, [0.35, 0.35, 0.30], bounds=[(0, 1), (0, 1), (0, 1)], method='SLSQP')
    opt_w = res.x / np.sum(res.x)
    print(f'Optimal Weights: LightGBM={opt_w[0]:.3f}, XGBoost={opt_w[1]:.3f}, CatBoost={opt_w[2]:.3f}')
    
    ensemble_oof = opt_w[0] * rank_lgb + opt_w[1] * rank_xgb + opt_w[2] * rank_cb
    print(f'-> Master Ensemble 5-Fold OOF AUC: {roc_auc_score(y, ensemble_oof):.6f}')
    
    # --- Output Generation & Verification ---
    final_pred = opt_w[0] * test_lgb + opt_w[1] * test_xgb + opt_w[2] * test_cb
    submission = pd.DataFrame({
        'SK_ID_CURR': test['SK_ID_CURR'].astype(int),
        'TARGET': final_pred
    })
    sub_path = OUTPUT_DIR / 'submission.csv'
    submission.to_csv(sub_path, index=False)
    print(f'\nSubmission written to: {sub_path} ({sub_path.stat().st_size / (1024*1024):.2f} MB)')
    
    # Assertions
    assert list(submission.columns) == list(sample_sub.columns), 'Columns mismatch!'
    assert len(submission) == len(test), 'Length mismatch!'
    assert submission['SK_ID_CURR'].equals(test['SK_ID_CURR']), 'ID ordering mismatch!'
    assert submission['TARGET'].notna().all(), 'NaN values found!'
    assert np.isfinite(submission['TARGET']).all(), 'Infinite values found!'
    assert submission['TARGET'].between(0, 1).all(), 'Probabilities outside [0, 1]!'
    print('ALL VERIFICATION ASSERTIONS PASSED!')
    print(f'Total Execution Time: {(time.time() - start_time) / 60:.2f} minutes')

if __name__ == '__main__':
    run_pipeline()
