import sys
sys.path.append("../../src")

import numpy as np
import pandas as pd
import lightgbm
from utils import ndcg_calculator



class LGBMRank():
    def __init__(self, 
                train_df:pd.DataFrame(), 
                model_params:dict={'n_estimators':5},
                path:str='../../data/',
                mode:str='week',
                n=25):

        self.train_df = train_df
        self.path = path
        self.model_params = model_params
        self.sample_sumbission_week = pd.read_parquet(path + 'sample_sumbission_week.parquet')
        self.sample_sumbission_month = pd.read_parquet(path + 'sample_sumbission_month.parquet')
        self.test_answer_week = pd.read_parquet(path + 'test_answer_week.parquet')
        self.test_answer_month = pd.read_parquet(path + 'test_answer_month.parquet')
        self.mode = mode

        self.n = n
        self.X_train, self.y_train, self.train_group = self.lgbm_preprocess(self.train_df)




    def lgbm_preprocess(self, train_df:pd.DataFrame()):
        X_train = train_df.drop(columns=['rating'])
        y_train = train_df['rating']
        
        train_group = train_df.groupby('profile_id')['profile_id'].count().to_numpy()
        return X_train, y_train, train_group




    def train(self):

        model_params = self.model_params

        model = lightgbm.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="dart",
            num_leaves= 20,
            learning_rate=0.005,
            n_estimators= model_params['n_estimators'],
            importance_type='gain',
            verbose= -1,
            random_state= 42
        )
        
        model.fit(
        X=self.X_train,
        y=self.y_train,
        group=self.train_group,
        )
        
        feature_importances_df = pd.DataFrame(dict(zip(self.X_train.columns, model.feature_importances_)), \
                                            index=['feature_importances']).T
        

        return model, feature_importances_df




    def valid_evaluation(self)->pd.DataFrame():
        
        X_train = self.X_train
        n = self.n
        


        model, feature_importances_df = self.train()
        print(feature_importances_df)

        pred = model.predict(X_train)
        X_train['pred'] = pred

        if self.mode == 'week':
            sample_sumbission = self.sample_sumbission_week
            test_answer = self.test_answer_week
            print('week performance')
        else:
            sample_sumbission = self.sample_sumbission_month
            test_answer = self.test_answer_month
            print('month performance')
        
        # each user pred 25 items
        lgbm_sub_df = X_train.sort_values(by='pred', ascending=False).groupby('profile_id').head(25)
        lgbm_user_items_dict = lgbm_sub_df.groupby('profile_id')['album_id'].unique().to_dict()
        sample_sumbission['album_id'] = sample_sumbission['profile_id'].apply(lambda x: lgbm_user_items_dict.get(x, []))

        print('lgbm ndcg:', ndcg_calculator(sample_sumbission, test_answer, n))
        
        return X_train, sample_sumbission