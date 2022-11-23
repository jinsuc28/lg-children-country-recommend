import numpy as np
import pandas as pd
import lightgbm
from utils import ndcg_calculator
import datetime as dt


def dataload(path: str = '../../data/'):
    test_answer_week = pd.read_parquet(path + "test_answer_week.parquet")
    test_answer_month = pd.read_parquet(path + "test_answer_month.parquet")

    df_train_week = pd.read_parquet(path + "train_week.parquet")
    df_train_month = pd.read_parquet(path + "train_month.parquet")

    sample_sumbission_week = pd.read_parquet(path + "sample_sumbission_week.parquet")
    sample_sumbission_month = pd.read_parquet(path + "sample_sumbission_month.parquet")

    df_train_week.sort_values(by='log_dt', inplace=True)
    df_train_month.sort_values(by='log_dt', inplace=True)

    return test_answer_week, test_answer_month, df_train_week, df_train_month, sample_sumbission_week, sample_sumbission_month

## Feature engineering
# week & day feature engineering
def week_day_feature(df_train: pd.DataFrame()) -> pd.DataFrame():
    df_train['week'] = df_train['log_date'].apply(lambda x: x.isocalendar()[1])
    df_train['day'] = df_train['log_date'].apply(lambda x: x.isocalendar()[2])
    week_min = df_train.week.min()
    df_train['week'] = df_train['week'].apply(lambda x: x - week_min)

    return df_train


# album_cnt & album_rank feature engineering
def album_cnt_rank_feature(df_train: pd.DataFrame()) -> pd.DataFrame():
    album_cnt = df_train.album_id.value_counts().reset_index().rename(
        columns={'index': 'album_id', 'album_id': 'album_cnt'})
    album_cnt['rank'] = album_cnt['album_cnt'].rank(method='first', ascending=False)
    df_train = df_train.merge(album_cnt, on='album_id')

    return df_train


def feature_engineering(df_train: pd.DataFrame()) -> pd.DataFrame():
    df_train = week_day_feature(df_train)
    df_train = album_cnt_rank_feature(df_train)

    return df_train


# MP@300 & MP_percent feature made
def MP_candidate(df_train: pd.DataFrame(), cand=300) -> pd.DataFrame():
    MP_df = df_train.album_id.value_counts().reset_index()
    MP_df.rename(columns={'index': 'album_id', 'album_id': 'item_cnt'}, inplace=True)

    # MP@300
    '''
    MP 중복 시청 포함 
    '''
    MP_cand = MP_df[:cand][['album_id']]
    return MP_cand


# latest 1 day each user@5
def latest_candidate(df_train: pd.DataFrame()) -> pd.DataFrame():
    seven_days = df_train['log_date'].max() - dt.timedelta(days=1)
    latest_history = df_train[df_train['log_date'] >= seven_days]
    latest_history = latest_history.groupby(['album_id']).count()['profile_id'].reset_index().rename(
        columns={'profile_id': 'latest_cnt'})

    # history count more than least 2
    latest_cand = latest_history[latest_history['latest_cnt'] >= 2][['album_id']].drop_duplicates()

    return latest_cand


def candidate(df_train: pd.DataFrame(), cand=300) -> pd.DataFrame():
    MP_cand = MP_candidate(df_train, cand)
    latest_cand = latest_candidate(df_train)
    cand = pd.concat([MP_cand, latest_cand])
    cand.drop_duplicates('album_id', inplace=True)
    cand['rating'] = 1

    return cand, MP_cand


def label_preprocess(df_train: pd.DataFrame(), cand: pd.DataFrame()):

    merge_train_week = df_train.drop_duplicates(subset=['profile_id', 'album_id'])
    train_df = pd.merge(merge_train_week, cand, how='left', on='album_id')

    drop_list = ['ss_id', 'act_target_dtl', 'payment', 'continuous_play', 'short_trailer', 'log_dt', 'log_date']
    train_df.drop(columns=drop_list, inplace=True)
    train_df.fillna(0, inplace=True)

    return train_df


def lgbm_preprocess(train_df: pd.DataFrame()):
    X_train = train_df.drop(columns=['rating'])
    y_train = train_df['rating']

    train_group = train_df.groupby('profile_id')['profile_id'].count().to_numpy()
    return X_train, y_train, train_group


def preprocess(train: pd.DataFrame(), cand: pd.DataFrame()):
    train_df = label_preprocess(train, cand)
    X_train, y_train, train_group = lgbm_preprocess(train_df)

    return X_train, y_train, train_group


def train(X_train:pd.DataFrame(), y_train:pd.Series(), train_group:np.array, model_params:dict=None):

    if model_params == None:
        model_params = model_params = {
                                        'n_estimators':100,
                                        'verbose':1,
                                        'random_state':42,
                                        'eval_at':25
                                        }

    model = lightgbm.LGBMRanker(
                                objective="lambdarank",
                                metric="ndcg",
                                boosting_type="dart",
                                n_estimators=model_params['n_estimators'],
                                importance_type='gain',
                                verbose=model_params['verbose'],
                                random_state=model_params['random_state']
                                )

    model.fit(
        X=X_train,
        y=y_train,
        group=train_group,
    )

    feature_importances_df = pd.DataFrame(dict(zip(X_train.columns, model.feature_importances_)), index=['feature_importances']).T

    return model, feature_importances_df


def valid_evaluation(
                    model,
                    X_train: pd.DataFrame(),
                    sample_sumbission: pd.DataFrame(),
                    n: int,
                    MP_cand: pd.DataFrame(),
                    test_answer
                    ) -> pd.DataFrame():

    pred = model.predict(X_train)
    X_train['pred'] = pred

    MP_list = MP_cand.album_id.values

    # each user pred 25 items
    lgbm_sub_df = X_train.sort_values(by='pred', ascending=False).groupby('profile_id').head(n)
    lgbm_user_items_dict = lgbm_sub_df.groupby('profile_id')['album_id'].unique().to_dict()
    sample_sumbission['album_id'] = sample_sumbission['profile_id'].apply(lambda x: lgbm_user_items_dict[x])

    # cold start user file MP_list top25
    sample_sumbission_cold = sample_sumbission.copy()
    sample_sumbission_cold['album_id'] = sample_sumbission_cold['album_id'] \
        .apply(lambda x: list(dict.fromkeys(np.append(x, MP_list)))[:n])

    print('lgbm ndcg:', ndcg_calculator(sample_sumbission, test_answer, n))
    print('lgbm ndcg cold_user to MP:', ndcg_calculator(sample_sumbission_cold, test_answer, n))

    return X_train, sample_sumbission, sample_sumbission_cold


def evaluation(
                X_train: pd.DataFrame(),
                n: int,
                MP_cand: pd.DataFrame(),
                submission_path: str = '../../data/'
                ) -> pd.DataFrame():

    submission = pd.read_csv(submission_path + 'sample_submission.csv')

    MP_list = MP_cand.album_id.values

    # each user pred 25 items
    lgbm_sub_df = X_train.sort_values(by='pred', ascending=False).groupby('profile_id').head(n)
    lgbm_user_items_dict = lgbm_sub_df.groupby('profile_id')['album_id'].unique().to_dict()
    submission['predicted_list'] = submission['profile_id'].apply(lambda x: lgbm_user_items_dict.get(x, []))

    # cold start user file MP_list top25
    submission_cold = submission.copy()
    submission_cold['predicted_list'] = submission_cold['predicted_list'].apply(lambda x: list(dict.fromkeys(np.append(x, MP_list)))[:n])

    # 제출 조건 충족 확인
    assert submission_cold.profile_id.nunique() == submission_cold.profile_id.nunique()
    for pred_list in submission_cold.predicted_list:
        assert len(pred_list) == n

    return submission, submission_cold