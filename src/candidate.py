import numpy as np
import pandas as pd
from tqdm import tqdm


def most_popular(df_train):
    MP_df = df_train.album_id.value_counts().head(50).reset_index()
    MP_df.columns = ['album_id','total_counts']
    MP_df['join_col'] = 1

    customer_df = df_train[['profile_id']].drop_duplicates()
    customer_df['join_col'] = 1

    MP_cand = customer_df.merge(MP_df, on='join_col').drop_duplicates(subset=['profile_id','album_id'])[['profile_id','album_id','total_counts']]

    return MP_cand



def general_most_popular(df_train):
    last_week_list = np.sort(df_train.week.unique())

    # 마지막 6,5주 각각 MP를 10개 뽑음
    last_week_ver1 = last_week_list[-1]
    last_week_ver2 = last_week_list[-2]

    # 마지막 6주
    MP_latest_ver1_df = df_train.query(f"week == {last_week_ver1}")

    MP_df = MP_latest_ver1_df.groupby('album_id')['profile_id'].count().sort_values(ascending=False)
    MP_df = MP_df.reset_index()
    MP_df.columns = ['album_id','counts']
    MP_candidate_df = MP_df[:10].copy()
    MP_candidate_df['join_col'] = 1

    # df_train_week 전체 유저 대상으로 후보군을 뽑을 것임
    customer_df = df_train[['profile_id']].drop_duplicates()
    customer_df['join_col'] = 1
    popular_articles_cand_ver1 = customer_df.copy()
    popular_articles_cand_ver1 = popular_articles_cand_ver1.merge(MP_candidate_df, on="join_col")

    popular_articles_cand_ver1.drop_duplicates(subset=['profile_id','album_id'],inplace=True)

    # 마지막 5주

    MP_latest_ver2_df = df_train.query(f"week == {last_week_ver2}")

    MP_df = MP_latest_ver2_df.groupby('album_id')['profile_id'].count().sort_values(ascending=False)
    MP_df = MP_df.reset_index()
    MP_df.columns = ['album_id','general_counts']
    MP_candidate_df = MP_df[:10].copy()
    MP_candidate_df['join_col'] = 1

    customer_df = df_train[['profile_id']].drop_duplicates()
    customer_df['join_col'] = 1
    popular_articles_cand_ver2 = customer_df.copy()
    popular_articles_cand_ver2 = popular_articles_cand_ver2.merge(MP_candidate_df, on="join_col")

    popular_articles_cand_ver2.drop_duplicates(subset=['profile_id','album_id'],inplace=True)

    # 마지막 6,5주 concat
    popular_articles_cand = pd.concat([popular_articles_cand_ver1, popular_articles_cand_ver2])
    popular_articles_cand = popular_articles_cand.groupby(['profile_id','album_id'])['general_counts'].sum().reset_index()

    general_MP_feature = popular_articles_cand[['album_id','general_counts']].drop_duplicates()

    return popular_articles_cand, general_MP_feature



def personal_most_popular(personal_train):
    personal_MP_df = personal_train.groupby(['profile_id','album_id'])[['ss_id']].count().reset_index()
    personal_MP_df.columns = ['profile_id','album_id','personal_counts']

    personal_MP_feature = personal_MP_df.copy()

    # 서로 다른날 5회 이상 시청한 앨범만
    personal_MP = personal_MP_df[personal_MP_df['personal_counts'] >= 5]
    personal_MP = personal_MP.sort_values(by=['profile_id','personal_counts'],ascending=False)

    # 상위 5개만 pick
    head_df_list = []
    # 전체 유저 대상으로 뽑기
    for user_id in tqdm(personal_train.profile_id.unique()):
        personal_MP_user_len = len(personal_MP[personal_MP['profile_id']==user_id].head())
        random_choice_list = personal_MP.album_id.unique()
        if personal_MP_user_len <5:
            # 5개 아이템이 없는 경우 랜덤으로 없는 개수만 만큼 choice
            user_df = personal_MP[personal_MP['profile_id']==user_id]
            df = pd.DataFrame()
            np.random.seed(42)
            random_choices = np.random.choice(random_choice_list, size=(5-personal_MP_user_len))
            df['profile_id'] = [user_id for _ in range(5-personal_MP_user_len)]
            df['album_id'] = random_choices
            df = pd.concat([user_df, df])
            head_df_list.append(df)
        else:
            head_df_list.append(personal_MP[personal_MP['profile_id']==user_id].head())
            
    personal_MP_candidate = pd.concat(head_df_list)

    return personal_MP_candidate, personal_MP_feature




def user_genre_most_popular(df_train, meta_df):

    # meta data load
    meta_df = meta_df[['album_id','genre_mid','run_time','cast_1','cast_2','cast_3']]
    meta_df = meta_df.drop_duplicates()

    df_train_meta = pd.merge(df_train,meta_df, how="left", on='album_id')
    
    # 유저 선호 장르별 MP
    user_genre_df = df_train_meta.groupby(['profile_id','genre_mid']).count()['ss_id'].reset_index()
    user_genre_df.columns = ['profile_id','genre_mid','genre_cnt']
    user_genre_df = user_genre_df.groupby(['profile_id','genre_mid']).sum().reset_index().sort_values(by=['profile_id','genre_cnt'],ascending=False)

    # 장르 선호도 피처 만들기
    ## 100이상 시청한 사람들만 percent
    user_total_watch_dict = user_genre_df.groupby('profile_id')['genre_cnt'].sum()\
                            [user_genre_df.groupby('profile_id')['genre_cnt'].sum()>=100].to_dict()
    # 전체 시청 피처 만들기
    user_genre_df['user_genre_cnt'] = user_genre_df['profile_id'].apply(lambda x: user_total_watch_dict.get(x, None))
    user_genre_df['user_genre_percent'] = user_genre_df['genre_cnt']/user_genre_df['user_genre_cnt']
    user_genre_df.drop(columns=['user_genre_cnt'],inplace=True)
    user_genre_df.dropna(subset=['user_genre_percent'],axis=0,inplace=True)

    # 장르별 탑 아이템 dict 담기
    genre_top_items = {}
    genre_count = df_train_meta['genre_mid'].value_counts()
    for genre in genre_count.index:
        genre_top_items[genre] = list(df_train_meta[df_train_meta['genre_mid']==genre]['album_id'].value_counts().head(10).index)


    # 모든 유저 선호 장르 MP candidate
    # 선호장르가 없을 경우 Top 장르인 "노래율동", "TV만화" 장르 MP candidate
    df_list = []
    for user_id in df_train.profile_id.unique():
        user_genres = user_genre_df[user_genre_df['profile_id']== user_id].head(2)['genre_mid']
        
        df = pd.DataFrame()
        if len(user_genres) == 0:
            
            df['album_id'] = genre_top_items['노래율동']
            df['album_id'] = genre_top_items['TV만화']
            
        elif len(user_genres) == 1:
            genre_list_1 = genre_top_items[user_genres.values[0]]
            genre_list_2 = genre_top_items['노래율동']
            df['album_id'] = list(dict.fromkeys(np.append(genre_list_1,genre_list_2)))
            
        elif len(user_genres) == 2:
            genre_list_1 = genre_top_items.get(user_genres.values[0],[])
            genre_list_2 = genre_top_items.get(user_genres.values[1],[])
            df['album_id'] = list(dict.fromkeys(np.append(genre_list_1,genre_list_2)))

        df['profile_id'] = user_id
        df_list.append(df)
        
    genre_candidate = pd.concat(df_list, ignore_index=True)
    genre_candidate = genre_candidate[['profile_id','album_id']]

    
    return genre_candidate


def age_MP(history_df, profile_df):
    age_df = pd.merge(history,profile,how="left",on="profile_id")
    age_mp_df = age_df.groupby(['age','album_id'])[["act_target_dtl"]].count().reset_index()
    age_mp_df = age_mp_df.rename(columns={"act_target_dtl":'age_album_counts'}).sort_values(by=["age","age_album_counts"],ascending=False)

    age_mp=[]
    for i in range(1,14):
        age = i
        album_ids = age_mp_df.loc[age_mp_df[(age_mp_df["age"]==i)].index[0:15]].album_id.values
        for j in album_ids : 
            album = j
            age_mp.append({"age":age , "album_id": album})
    age_mp_cand_df = pd.DataFrame(age_mp)
    
    for i in range(1,14):
        if i == 1 :
            age_pool = age_mp_cand_df[age_mp_cand_df["age"]<=i+2] 
            age_pool['age']= age_pool['age'].replace([i+1,i+2],i) 
            age_pool_mp_df = age_pool
        if i == 13 :
            age_pool = age_mp_cand_df[age_mp_cand_df["age"]<=i-2]
            age_pool['age']= age_pool['age'].replace([i-1,i-2],i) 
            age_pool_mp_df = pd.concat([age_pool_mp_df,age_pool])
        else :
            age_pool = age_mp_cand_df[(age_mp_cand_df["age"]>=i-1)&(age_mp_cand_df["age"]<=i+1)] 
            age_pool['age']= age_pool['age'].replace([i-1,i+1],i) 
            age_pool_mp_df = pd.concat([age_pool_mp_df,age_pool])
            
    age_pool_mp_df = age_pool_mp_df.drop_duplicates()  
    add_proID = history.merge(profile,how="left",on="profile_id").drop(columns="album_id")
    add_proID = add_proID.merge(age_mp_cand_df,how="left",on="age")
    age_MP_candidate = add_proID[["profile_id","album_id"]]
    age_MP_candidate = age_MP_candidate.drop_duplicates()
    
    return age_MP_candidate