#!/usr/bin/python
# -*- coding=utf-8 -*-

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing

train_df = pd.read_csv('./data/train.csv', parse_dates=['timestamp'])
test_df = pd.read_csv('./data/test.csv', parse_dates=['timestamp'])

#处理离散值
def refine_object(train_df):
    for f in train_df.columns:
        if train_df[f].dtype == 'object':
            lencode = preprocessing.LabelEncoder()
            lencode.fit(list(train_df[f].values))
            train_df[f] = lencode.transform(list(train_df[f].values))

refine_object(train_df)
refine_object(test_df)

train_y = train_df.price_doc.values
#test_y = test_df.price_doc.values

train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
test_X = test_df.drop(["id", "timestamp"], axis=1)

xgb_params = {
        'eta': 0.5,
        'max_depth': 9,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse', #评估函数
        'silent':1
}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
dtest = xgb.DMatrix(test_X, feature_names=test_X.columns.values)

model = xgb.train(xgb_params, dtrain, num_boost_round=100)
ypred = model.predict(dtest)
with open('predict.txt', 'w') as f:
    for itme in ypred:
        f.write("%s\n" % (itme))


