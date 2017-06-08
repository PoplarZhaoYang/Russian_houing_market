#!/usr/bin/python
# -*- coding=utf-8 -*-
"""训练得到xgboost的模型，并做CV调节参数
"""
import os
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import  cross_val_score 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer



class MyXgb(object):
    """训练xgboost模型
    * 直接调用train_model()方法训练
    """
    xgb_params = {}
    def __init__(self):
        self.xgb_params = {
            'eta': 0.5,
            'max_depth': 7,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'objective': 'reg:linear',
            'eval_metric': 'rmse',  # 评估函数
            'silent': 0,  # 显示xgboost训练过程
            'num_boost_round': 40
        }
    def refine_object(self, train_df):
        """处理类别对象为可训练数值特征
        """
        for _f in train_df.columns:
            if train_df[_f].dtype == 'object':
                lencode = preprocessing.LabelEncoder()
                lencode.fit(list(train_df[_f].values))
                train_df[_f] = lencode.transform(list(train_df[_f].values))
    def tune_model_param(self, source_file='./data/train.csv' ,aim_file='./tmp/xgb_param.p'):
        """sklearn api
        """
        train_df = pd.read_csv(source_file, parse_dates=['timestamp'])
        self.refine_object(train_df)
        train_y = train_df.price_doc.values
        train_x = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

        model = xgb.XGBRegressor(objective='reg:linear')
        param_grids = {
            'max_depth': np.arange(3, 13, 2),
            'learning_rate': np.arange(0.01, 0.2, 0.02),
            'n_estimators': np.arange(100, 500, 100),
            'subsample': np.arange(0.3, 0.8, 0.1),
            'min_child_weight': [1, 2, 3],
            'reg_lambda': [0.01, 0.1, 1, 10, 100]
        }
        score_func = make_scorer(self.score_RMSLE, greater_is_better=False)
        clf = GridSearchCV(model, param_grid=param_grids, cv=5, n_jobs=7,
                            verbose=4, scoring='neg_mean_squared_error')
        clf.fit(train_x, train_y)
        print clf.best_params_
        print clf.best_score_
        with open(aim_file, 'w') as _f:
            pickle.dump(clf.best_params_, _f)
    def train_model(self, source_data='./data/train.csv', source_param='./tmp/xgb_param.p',
                     model_file='./tmp/xgb.model', retrain=False):
        """传入训练数据的文件名，和保存地址文件名
        * retrain参数说明是否重新训练模型
        """
        train_df = pd.read_csv(source_data, parse_dates=['timestamp'])
        self.refine_object(train_df)
        train_y = train_df.price_doc.values
        train_x = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

        xgb_param = {}
        with open(source_param, 'r') as _f:
            xgb_param = pickle.load(_f)
        
        model = xgb.XGBRegressor( max_depth=xgb_param['max_depth'],
                               #  learning_rate=xgb_param['learning_rate'],
                               #  n_estimators=xgb_param['n_estimators'],
                               #  subsample=xgb_param['subsample'],
                               #  min_child_weight=xgb_param['min_child_weight'],
                               #  reg_lambda=xgb_param['reg_lambda'],
                                 objective='reg:linear',
                                 n_jobs=7,
                                 )
        model.fit(train_x, train_y)
        with open(model_file, 'w') as _f:
            print "成功存储了训练模型于'/tmp/xgb.model'"
            pickle.dump(model, _f)
    
    def predict_price(self, test_file='./data/test.csv', model_file='./tmp/xgb.model'):
        test_df = pd.read_csv(test_file, parse_dates=['timestamp'])
        test_id = test_df["id"]
        self.refine_object(test_df)
        test_x = test_df.drop(["id", "timestamp"], axis=1)

        with open(model_file, 'r') as _f:
            model = pickle.load(_f)
        
        predicted = model.predict(test_x)
        with open('predict.txt', 'w') as f:
            f.write('id,price_doc\n')
            for ids, itme in zip(test_id, predicted):
                f.write("%d,%.2f\n" % (ids, itme))
    def local_cv(self, train_file='./data/train.csv', model_file='./tmp/xgb.model'):
        """本地交叉验证分数
        """
        with open(model_file, 'r') as _f:
            model = pickle.load(_f)
        train_df = pd.read_csv(train_file, parse_dates=['timestamp'])
        self.refine_object(train_df)
        train_y = train_df.price_doc.values
        train_x = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
        score_func = make_scorer(self.score_RMSLE, greater_is_better=False)
        scores = cross_val_score(model, train_x, train_y, cv=5, scoring=score_func)
        print "RMSLE %.2f (+/- %.2f)" % (scores.mean(), scores.std())
    def score_RMSLE(self, ground_true, prediction):
        ground_true1 = ground_true + 1
        prediciton1 = prediction + 1
        ground_true_log = np.log(ground_true1)
        prediction_log = np.log(prediciton1)
        temp = (ground_true_log - prediction_log) ** 2
        sum_temp = np.sum(temp) / ground_true.shape[0]
        return np.sqrt(sum_temp)

def main():
    """main
    """
    T = MyXgb()
    #T.tune_model_param()
   # T.train_model()
   # T.local_cv()
    #T.predict_price()

if __name__ == "__main__":
    main()
"""sklearn API
print scoring.mean(), scoring.std()
"""
