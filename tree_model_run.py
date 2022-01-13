import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np


data_dir = 'wsdm_model_data/'
# data = pd.read_csv(data_dir + "train_data_tree.txt", sep="\t")

train_df = pd.read_csv(data_dir + "train_data_tree.txt", sep="\t").reset_index(drop=True)
test_df = pd.read_csv(data_dir + "test_data_tree.txt", sep="\t").reset_index(drop=True)

drop_feature_list = ['end_date','launch_seq','label','user_id']#+[x for x in train_df.columns if 'count_' in x]
label = 'label'
feature = [x for x in train_df.columns if x not in drop_feature_list]

#lgb
def lgb_custom_metric(y_true,y_pre):
    y_pre = y_pre.get_label()
    score = 100*(1-np.mean((np.abs(y_pre.reshape(-1) - y_true.reshape(-1))/7)))
    return 'self-metric', score, True
#自己用的
def custom_metric(y_true,y_pre):
    score = 100*(1-np.mean((np.abs(y_pre.reshape(-1) - y_true.reshape(-1))/7)))
    return score


n_fold = 5
# lgb参数
lgb_params = {
    "boosting_type": "gbdt",
    "objective": "mae",
    "learning_rate": 0.01,
    "max_depth": 6,
    "num_leaves": 15,
    "nthread": -1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}
y_val = np.zeros((train_df.shape[0]))
y_test = np.zeros((test_df.shape[0]))
score_list = []
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
# StratifiedKFold，KFold有什么区别？
for train_index, valid_index in skf.split(train_df[feature], train_df[label]):
    x_train, x_valid, y_train, y_valid = train_df[feature].iloc[train_index], train_df[feature].iloc[valid_index], \
                                         train_df[label].iloc[train_index], train_df[label].iloc[valid_index]

    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_valid, label=y_valid)

    model = lgb.train(lgb_params, train_data, valid_sets=[valid_data], num_boost_round=1000, verbose_eval=50,
                      early_stopping_rounds=50, feval=lgb_custom_metric)

    y_val[valid_index] = model.predict(x_valid)
    y_test += np.array(model.predict(test_df[feature]) / n_fold)

cv_score = custom_metric(y_val, train_df[label].values)
print(f'local cv: {cv_score}')

#特征重要性
feature_imp_df = pd.DataFrame()
feature_imp_df['fea_name'] = model.feature_name()
feature_imp_df['fea_imp'] = model.feature_importance()
feature_imp_df = feature_imp_df.sort_values('fea_imp',ascending=False)
print(feature_imp_df)

# cat
n_fold = 5
y_val_cat = np.zeros((train_df.shape[0]))
y_test_cat = np.zeros((test_df.shape[0]))
score_list = []
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
# StratifiedKFold，KFold有什么区别？
for train_index, valid_index in skf.split(train_df[feature], train_df[label]):
    x_train, x_valid, y_train, y_valid = train_df[feature].iloc[train_index], train_df[feature].iloc[valid_index], \
                                         train_df[label].iloc[train_index], train_df[label].iloc[valid_index]

    train_pool = Pool(x_train, y_train)
    eval_pool = Pool(x_valid, y_valid)

    cbt_model = CatBoostRegressor(iterations=10000,
                                  learning_rate=0.03,
                                  eval_metric='RMSE',
                                  loss_function='MAE',
                                  max_depth=6,
                                  use_best_model=True,
                                  random_seed=42,
                                  logging_level='Verbose',
                                  task_type='GPU',
                                  l2_leaf_reg = 0.5,
                                  # devices='0',
                                  # gpu_ram_part=0.5,
                                  early_stopping_rounds=50,

                                  )
    cbt_model.fit(train_pool, eval_set=eval_pool, verbose=50)

    y_val_cat[valid_index] = cbt_model.predict(x_valid)
    y_test_cat += np.array(cbt_model.predict(test_df[feature]) / n_fold)

cat_cv_score = custom_metric(y_val_cat, train_df[label].values)
print(f'local cv: {cat_cv_score}')

feature_imp_cat_df = pd.DataFrame()
feature_imp_cat_df['fea_name'] = cbt_model.feature_names_
feature_imp_cat_df['fea']=cbt_model.feature_importances_
feature_imp_cat_df = feature_imp_cat_df.sort_values(['fea'], ascending=False)
print(feature_imp_cat_df)



def handle_result(x):
    if x<=0:
        return 0
    elif x>=7:
        return 7
    else:
        return x

res = pd.DataFrame()
res['user_id'] = test_df['user_id']
res['pre'] = (y_test + y_test_cat) / 2
res['pre'] = res['pre'].apply(handle_result)
res.to_csv(f"tree_base_{cv_score}.csv", index=False, header=False, float_format="%.2f")