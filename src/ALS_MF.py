import pandas as pd
import numpy as np
import implicit
from implicit.als import AlternatingLeastSquares as ALS
from utils import ndcg_calculator
import scipy.sparse as sp

class MF():
    def __init__(self,
                train:pd.DataFrame(),
                clf={'factors':200 , 'regularization':0.005, 'iterations':3, 'n':100}):

        self.train = train.drop_duplicates(subset=['profile_id','album_id','ss_id'])
        self.userIdToIndex, self.indexToUserId, self.PdIdToIndex, self.indexToPdId, self.purchase_sparse = self.matrix()
        self.clf = clf

    def matrix(self):
        train = self.train
    
        PdIds = train.album_id.unique()

        PdIdToIndex = {}
        indexToPdId = {}

        colIdx = 0

        for PdId in PdIds:
            PdIdToIndex[PdId] = colIdx
            indexToPdId[colIdx] = PdId
            colIdx += 1
            
        userIds = train.profile_id.unique()

        userIdToIndex = {}
        indexToUserId = {}

        rowIdx = 0

        for userId in userIds:
            userIdToIndex[userId] = rowIdx
            indexToUserId[rowIdx] = userId
            rowIdx += 1

        rows = []
        cols = []
        vals = []

        for row in train.itertuples():
            rows.append(userIdToIndex[row.profile_id])
            cols.append(PdIdToIndex[row.album_id])
            vals.append(1)

        purchase_sparse = sp.csr_matrix((vals, (rows, cols)), shape=(rowIdx,colIdx))

        return userIdToIndex, indexToUserId, PdIdToIndex, indexToPdId, purchase_sparse

    def mf_train(self):
        train = self.train
        userIdToIndex = self.userIdToIndex
        indexToPdId = self.indexToPdId
        indexToUserId = self.indexToUserId
        purchase_sparse = self.purchase_sparse

        als_model = ALS(factors=self.clf['factors'], regularization=self.clf['regularization'], iterations = self.clf['iterations'], random_state=42)
        als_model.fit(purchase_sparse, show_progress=True)

        als_predict_list = []
        for user_id in train['profile_id'].unique():
            result = als_model.recommend(userIdToIndex[user_id], purchase_sparse[userIdToIndex[user_id]], N=self.clf['n'],filter_already_liked_items=False)
            als_predict_list.append(pd.DataFrame({'profile_id':[user_id for _ in range(len(result[0]))] ,'album_id':[indexToPdId[num] for num in result[0]]}))
        pred_df = pd.concat(als_predict_list)
        
        print("item vector shape:", als_model.item_factors.shape, "user vector shape", als_model.user_factors.shape)

        item_vector = pd.DataFrame(als_model.item_factors)
        user_vector = pd.DataFrame(als_model.user_factors)
        item_vector.columns = ["item_vector" + str(num) for num in item_vector.columns]
        user_vector.columns = ["user_vector" + str(num) for num in user_vector.columns]

        item_vector_feature = item_vector.reset_index()
        item_vector_feature["index"] = item_vector_feature["index"].apply(lambda x: indexToPdId[x])
        item_vector_feature = item_vector_feature.rename(columns={"index":"album_id"})

        user_vector_feature = user_vector.reset_index()
        user_vector_feature["index"] = user_vector_feature["index"].apply(lambda x: indexToUserId[x])
        user_vector_feature = user_vector_feature.rename(columns={"index":"profile_id"})


        return pred_df, item_vector_feature, user_vector_feature, als_model


class ReplayMF():
    def __init__(self,
                train:pd.DataFrame(),
                df_user_replay_bool:pd.DataFrame(),
                clf={'factors':200 , 'regularization':0.005, 'iterations':3, 'n':100}):

        self.train = train.drop_duplicates(subset=['profile_id','album_id','ss_id'])
        self.userIdToIndex, self.indexToUserId, self.PdIdToIndex, self.indexToPdId, self.purchase_sparse = self.matrix()
        self.clf = clf
        self.df_user_replay_bool = df_user_replay_bool

    def matrix(self):
        train = self.train

        PdIds = train.album_id.unique()

        PdIdToIndex = {}
        indexToPdId = {}

        colIdx = 0

        for PdId in PdIds:
            PdIdToIndex[PdId] = colIdx
            indexToPdId[colIdx] = PdId
            colIdx += 1

        userIds = train.profile_id.unique()

        userIdToIndex = {}
        indexToUserId = {}

        rowIdx = 0

        for userId in userIds:
            userIdToIndex[userId] = rowIdx
            indexToUserId[rowIdx] = userId
            rowIdx += 1

        rows = []
        cols = []
        vals = []

        for row in train.itertuples():
            rows.append(userIdToIndex[row.profile_id])
            cols.append(PdIdToIndex[row.album_id])
            vals.append(1)

        purchase_sparse = sp.csr_matrix((vals, (rows, cols)), shape=(rowIdx,colIdx))

        return userIdToIndex, indexToUserId, PdIdToIndex, indexToPdId, purchase_sparse

    def mf_train(self):
        train = self.train
        userIdToIndex = self.userIdToIndex
        indexToPdId = self.indexToPdId
        indexToUserId = self.indexToUserId
        purchase_sparse = self.purchase_sparse
        df_user_replay_bool = self.df_user_replay_bool
        df_user_replay_bool_dict = df_user_replay_bool.set_index("profile_id").to_dict()

        als_model = ALS(factors=self.clf['factors'], regularization=self.clf['regularization'], iterations = self.clf['iterations'], random_state=42)
        als_model.fit(purchase_sparse, show_progress=True)

        als_predict_list = []
        for user_id in train['profile_id'].unique():
            if df_user_replay_bool_dict["replay"][user_id]:
                result = als_model.recommend(userIdToIndex[user_id], purchase_sparse[userIdToIndex[user_id]], N=self.clf['n'],filter_already_liked_items=False)
                als_predict_list.append(pd.DataFrame({'profile_id':[user_id for _ in range(len(result[0]))] ,'album_id':[indexToPdId[num] for num in result[0]]}))
            else:
                result = als_model.recommend(userIdToIndex[user_id], purchase_sparse[userIdToIndex[user_id]], N=self.clf['n'])
                als_predict_list.append(pd.DataFrame({'profile_id':[user_id for _ in range(len(result[0]))] ,'album_id':[indexToPdId[num] for num in result[0]]}))

        pred_df = pd.concat(als_predict_list)

        print("item vector shape:", als_model.item_factors.shape, "user vector shape", als_model.user_factors.shape)

        item_vector = pd.DataFrame(als_model.item_factors)
        user_vector = pd.DataFrame(als_model.user_factors)
        item_vector.columns = ["item_vector" + str(num) for num in item_vector.columns]
        user_vector.columns = ["user_vector" + str(num) for num in user_vector.columns]

        item_vector_feature = item_vector.reset_index()
        item_vector_feature["index"] = item_vector_feature["index"].apply(lambda x: indexToPdId[x])
        item_vector_feature = item_vector_feature.rename(columns={"index":"album_id"})

        user_vector_feature = user_vector.reset_index()
        user_vector_feature["index"] = user_vector_feature["index"].apply(lambda x: indexToUserId[x])
        user_vector_feature = user_vector_feature.rename(columns={"index":"profile_id"})


        return pred_df, item_vector_feature, user_vector_feature