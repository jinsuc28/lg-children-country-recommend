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
import gc
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import logging


## load data
test_answer_week = pd.read_parquet("../input/lg-train-test/test_answer_week.parquet")
test_answer_month = pd.read_parquet("../input/lg-train-test/test_answer_month.parquet")

df_train_week = pd.read_parquet("../input/lg-train-test/train_week.parquet")
df_train_month = pd.read_parquet("../input/lg-train-test/train_month.parquet")

sample_sumbission_week = pd.read_parquet("../input/lg-train-test/sample_sumbission_week.parquet")
sample_sumbission_month = pd.read_parquet("../input/lg-train-test/sample_sumbission_month.parquet")

train_week = df_train_week.copy()
train_month = df_train_month.copy()
mf_sumbission_week = sample_sumbission_week.copy()
mf_sumbission_month = sample_sumbission_month.copy()

## you also need full dataset, if want to make submit.csv
history_mf = pd.read_csv("../input/lgground/history_data.csv")
sub=pd.read_csv('../input/lgground/sample_submission.csv')


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


def train(csr_train, factors=200, iterations=3, regularization=0.01, show_progress=True):
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



###########################
## MF with whole History
###########################

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def extract_data(base_path):
    df_history = pd.read_csv("")
    logger.info("History are extracted")
    return df_history


def preprocess_date(df_history):
    ## 날짜 전처리
    df_history = df_history.assign(log_dt = pd.to_datetime(df_history.log_time//100, format="%Y%m%d%H%M"))
    df_history = df_history.assign(log_date = df_history.log_dt.dt.floor("D"))
    df_history = df_history.drop("log_time", axis=1)
    logger.info("Datetime preprocess completed")
    return df_history

def real_submit(model, csr_train, sub):  
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
            album_ids = [item_ids[item_id] for item_id in user_items] 
            pred_df.append({'profile_id':profile_id,'predicted_list':album_ids})
    sub = pd.DataFrame(pred_df)    
    return sub

user_item_maps(history_mf)
mf_csr = make_csr_matrix(history_mf)
mf_model = train(mf_csr)
mf_preds = real_submit(mf_model, mf_csr, sub)


# week best Factors: 1000 - Iterations:  3 - Regularization: 0.100 ==> LB 0.2266 / ndcg 0.11888 
''' 현재 최고 성능 (week valid 기준)
### factors=200, iterations=3, regularization=0.05  ==> LB 0.2275 / ndcg 0.11312
    정확히 2배의 결과. 오히려 자체측정 최고 param보다 잘 나왔다.
    factor가 많다고 좋은 결과가 나오는 것은 아니다.
'''

mf_preds['predicted_list'] = mf_preds['predicted_list'].apply(lambda x: str(x))
mf_preds.to_csv('als.csv', index=False)