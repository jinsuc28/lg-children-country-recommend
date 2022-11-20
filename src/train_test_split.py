from datetime import timedelta
from dateutil.relativedelta import relativedelta

import logging
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def extract_data(base_path):

    df_submit = pd.read_csv(base_path + "sample_submission.csv")
    df_history = pd.read_csv(base_path + "history_data.csv")
    logger.info("History and Submit are extracted")

    return df_submit, df_history


def preprocess_date(df_history):

    ## 날짜 전처리
    df_history = df_history.assign(log_dt = pd.to_datetime(df_history.log_time//100, format="%Y%m%d%H%M"))
    df_history = df_history.assign(log_date = df_history.log_dt.dt.floor("D"))
    df_history = df_history.drop("log_time", axis=1)
    logger.info("Datetime preprocess completed")

    return df_history


def split_history(df_history):
    last_date = df_history.log_date.max()

    ## 1주일 전, 1달 전
    last_week = last_date - timedelta(days=7)
    last_month = last_date - relativedelta(months=1)

    df_train_week = df_history[df_history.log_date <= last_week]
    df_test_week = df_history[df_history.log_date > last_week]
    df_train_month = df_history[df_history.log_date <= last_month]
    df_test_month = df_history[df_history.log_date > last_month]

    assert df_test_week.log_date.nunique() == 7
    assert df_test_month.log_date.nunique() == 31

    logger.info("Splitting history data is completed")

    return df_train_week, df_test_week, df_train_month, df_test_month


def answer_generate(df_train_week, df_test_week, df_train_month, df_test_month):
    ## 최근 25개만
    df_test_week = df_test_week.sort_values("log_dt")
    df_test_month = df_test_month.sort_values("log_dt")

    ## Train에 있는 profile/album만 test에 두기

    logger.info("train에 있는 profile/album만 남기기")

    logger.info(f"Test data size:{len(df_test_week):,}(week)")
    df_test_week = df_test_week.merge(df_train_week[["profile_id", "album_id"]].drop_duplicates(),
                                      on=["profile_id", "album_id"])
    logger.info(f"-> Test data size:{len(df_test_week):,}(week)")

    logger.info(f"Test data size:{len(df_test_month):,}(month)")
    df_test_month = df_test_month.merge(df_train_month[["profile_id", "album_id"]].drop_duplicates(),
                                        on=["profile_id", "album_id"])
    logger.info(f"-> Test data size:{len(df_test_month):,}(month)")

    test_answer_week = (
        df_test_week
        .drop_duplicates(["profile_id", "album_id"])
        .groupby("profile_id").head(25)
        .groupby("profile_id").album_id.apply(list)
        .reset_index()
    )

    test_answer_month = (
        df_test_month
        .drop_duplicates(["profile_id", "album_id"])
        .groupby("profile_id").head(25)
        .groupby("profile_id").album_id.apply(list)
        .reset_index()
    )

    assert test_answer_week.album_id.apply(len).max() == 25
    assert test_answer_month.album_id.apply(len).max() == 25

    logger.info("Answer data generated")

    return test_answer_week, test_answer_month

def generate_sample_submission(test_answer_week, test_answer_month):

    sample_sumbission_week = test_answer_week.copy()
    sample_sumbission_week.album_id = [[]] * len(sample_sumbission_week)

    sample_sumbission_month = test_answer_month.copy()
    sample_sumbission_month.album_id = [[]] * len(sample_sumbission_month)

    logger.info("Sample submission data generated")

    return sample_sumbission_week, sample_sumbission_month

def load_data(
        base_path,
        test_answer_week, test_answer_month,
        df_train_week, df_train_month,
        sample_sumbission_week, sample_sumbission_month
):
    test_answer_week.to_parquet(base_path+"test_answer_week.parquet")
    test_answer_month.to_parquet(base_path+"test_answer_month.parquet")

    df_train_week.to_parquet(base_path+"train_week.parquet")
    df_train_month.to_parquet(base_path+"train_month.parquet")

    sample_sumbission_week.to_parquet(base_path+".sample_sumbission_week.parquet")
    sample_sumbission_month.to_parquet(base_path+".sample_sumbission_month.parquet")

    logger.info(f"All data are loaded at {base_path}")

def execute():
    base_path = "../data/"

    df_submit, df_history = extract_data(base_path)


    df_history = preprocess_date(df_history)

    df_train_week, df_test_week, df_train_month, df_test_month = split_history(df_history)
    test_answer_week, test_answer_month = answer_generate(df_train_week, df_test_week, df_train_month, df_test_month)
    sample_sumbission_week, sample_sumbission_month = generate_sample_submission(test_answer_week, test_answer_month)

    load_data(
        base_path,
        test_answer_week, test_answer_month,
        df_train_week, df_train_month,
        sample_sumbission_week, sample_sumbission_month
    )


if __name__ == "__main__":
    execute()