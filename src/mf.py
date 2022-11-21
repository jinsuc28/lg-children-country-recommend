import sys
sys.path.append("../../src")
from scipy.sparse import csr_matrix
import implicit
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import ndcg_calculator


def user_item_maps(df):
    global ALL_USERS, ALL_ITEMS, user_ids, item_ids, user_map, item_map
    ALL_USERS = df['profile_id'].unique().tolist()
    ALL_ITEMS = df['album_id'].unique().tolist()

    user_ids = dict(list(enumerate(ALL_USERS)))
    item_ids = dict(list(enumerate(ALL_ITEMS)))

    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    df['profile_id'] = df['profile_id'].map(user_map)
    df['album_id'] = df['album_id'].map(item_map)
    return ALL_USERS, ALL_ITEMS, user_ids, item_ids, user_map, item_map

def make_csr_matrix(df):
    row = df['profile_id'].values
    col = df['album_id'].values
    data = np.ones(df.shape[0])
    csr_train = csr_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
    return csr_train


def train(csr_train, factors=200, iterations=5, regularization=0.01, show_progress=True):
    model = implicit.als.AlternatingLeastSquares(factors=factors, 
                                                 iterations=iterations, 
                                                 regularization=regularization, 
                                                 random_state=42)
    model.fit(csr_train, show_progress=show_progress)
    return model


def submit(model, csr_train, sample_sumbission_week):  # month 돌릴 때는 혼동 없도록 인자 잘 전달하기
    preds = []
    batch_size = 2000
    to_generate = np.arange(len(ALL_USERS))
    pred_df = []
    for startidx in range(0, len(to_generate), batch_size):
        batch = to_generate[startidx : startidx + batch_size]
        ids, scores = model.recommend(batch, csr_train[batch], N=25, filter_already_liked_items=False)
        for i, profile_id in enumerate(batch):
            profile_id = user_ids[profile_id]
            user_items = ids[i]
            album_ids = [item_ids[item_id] for item_id in user_items] #
            pred_df.append({'profile_id':profile_id,'album_id':album_ids})

    pred_dfs = pd.DataFrame(pred_df)
    sample_sumbission_week.drop(columns='album_id', inplace=True)
    sample_sumbission_week = sample_sumbission_week.merge(pred_dfs, on='profile_id')
    
    return sample_sumbission_week



## MF Week
user_item_maps(train_week)
week_csr = make_csr_matrix(train_week)
week_model = train(week_csr)
week_preds = submit(week_model, week_csr, sample_sumbission_week)


## MF month
user_item_maps(train_month)
month_csr = make_csr_matrix(train_month)
month_model = train(month_csr)
month_preds = submit(month_model, month_csr, sample_sumbission_month)


# Evaluation
mf_week_ndcg = ndcg_calculator(test_answer_week, week_preds, n)
mf_month_ndcg = ndcg_calculator(test_answer_month, month_preds, n)


print("Week performance")
print(f"nDCG(MF_week): {mf_week_ndcg:.4f}")

print("Month performance")
print(f"nDCG(MF_month): {mf_month_ndcg:.4f}")


'''
>> Next step
Neg sampling & pyper parameter tuning
'''