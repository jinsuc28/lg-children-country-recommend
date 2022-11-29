import sys
import numpy as np
import pandas as pd
from datetime import timedelta
import math



def dataload(path:str='../../data/'):

    test_answer_week = pd.read_parquet(path + "test_answer_week.parquet")
    test_answer_month = pd.read_parquet(path + "test_answer_month.parquet")

    df_train_week = pd.read_parquet(path + "train_week.parquet")
    df_train_month = pd.read_parquet(path + "train_month.parquet")

    sample_sumbission_week = pd.read_parquet(path + "sample_sumbission_week.parquet")
    sample_sumbission_month = pd.read_parquet(path + "sample_sumbission_month.parquet")

    df_train_week.sort_values(by='log_dt', inplace=True)
    df_train_month.sort_values(by='log_dt', inplace=True)
    
    return test_answer_week, test_answer_month, df_train_week, df_train_month, sample_sumbission_week, sample_sumbission_month




# week & day feature engineering
def day_feature(df_train:pd.DataFrame())->pd.Series():
    dates = df_train.log_date
    unique_dates = df_train.log_date.unique()
    unique_dates = np.sort(unique_dates)
    number_range = np.arange(len(unique_dates))
    date_number_dict = dict(zip(unique_dates, number_range))

    all_day_numbers = dates.map(date_number_dict)
    all_day_numbers = all_day_numbers.astype("int16")
    print('log date min:', dates.min(), 'log date max:', dates.max())
    print('min day:', all_day_numbers.min(), 'max day:', all_day_numbers.max())

    return all_day_numbers

def week_feature(df_train:pd.DataFrame())->pd.Series:
    pd_dates = df_train.log_date
    unique_dates = pd.Series(df_train.log_date.unique())
    numbered_days = unique_dates - unique_dates.min() + timedelta(1)
    numbered_days = numbered_days.dt.days
    extra_days = numbered_days.max() % 7
    numbered_days -= extra_days
    day_weeks = (numbered_days / 7).apply(lambda x: math.ceil(x))
    day_weeks_map = pd.DataFrame({"day_weeks": day_weeks, "unique_dates": unique_dates})\
                                                        .set_index("unique_dates")["day_weeks"]
    all_day_weeks = pd_dates.map(day_weeks_map)
    all_day_weeks = all_day_weeks.astype("int8")
    print('min week:', all_day_weeks.min(), 'max week:', all_day_weeks.max())

    return all_day_weeks

def day_week_feature_engineering(df_train:pd.DataFrame())->pd.DataFrame():
    all_day_numbers = day_feature(df_train)
    all_day_weeks = week_feature(df_train)
    
    df_train['day'] = all_day_numbers
    df_train['week'] = all_day_weeks

    return df_train


# train, label split
def train_label_split(df_train:pd.DataFrame())->pd.DataFrame():
    last_week = df_train['week'].max()
    print('split last week:', last_week)

    label_df = df_train.query(f'week=={last_week}')[['profile_id','album_id']]
    label_df.drop_duplicates(subset=['profile_id','album_id'],inplace=True)
    label_df['rating'] = 1

    df_train = df_train.query(f"week <= {last_week}")
    
    return df_train, label_df


def feature_dataload(path):
    watch_df = pd.read_csv(path + "watch_e_data.csv")
    buy_df = pd.read_csv(path + "buy_data.csv")
    search_df = pd.read_csv(path + "search_data.csv")
    meta_df = pd.read_csv(path + "meta_data.csv")
    profile_df = pd.read_csv(path + "profile_data.csv")

    return watch_df, buy_df, search_df, meta_df, profile_df


