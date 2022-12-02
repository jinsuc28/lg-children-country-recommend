import numpy as np
import pandas as pd
from tqdm import tqdm


def most_popular(df_train):
    MP_df = df_train.album_id.value_counts().reset_index()
    MP_df.columns = ['album_id','total_counts']
    MP_feature = MP_df.copy()
    MP_df['join_col'] = 1

    MP_df = MP_df.head(50)

    customer_df = df_train[['profile_id']].drop_duplicates()
    customer_df['join_col'] = 1

    MP_cand = customer_df.merge(MP_df, on='join_col').drop_duplicates(subset=['profile_id','album_id'])[['profile_id','album_id']]

    return MP_cand, MP_feature



def general_most_popular(df_train):
    last_week_list = np.sort(df_train.week.unique())

    # 마지막 6,5주 각각 MP를 10개 뽑음
    last_week_ver1 = last_week_list[-1]
    last_week_ver2 = last_week_list[-2]

    # 마지막 6주
    MP_latest_ver1_df = df_train.query(f"week == {last_week_ver1}")

    MP_df = MP_latest_ver1_df.groupby('album_id')['profile_id'].count().sort_values(ascending=False)
    MP_df = MP_df.reset_index()
    MP_df.columns = ['album_id','general_counts_ver1']
    MP_ver1 = MP_df.copy()
    MP_candidate_df = MP_df[:10].copy()
    MP_candidate_df['join_col'] = 1

    # df_train_week 전체 유저 대상으로 후보군을 뽑을 것임
    customer_df = df_train[['profile_id']].drop_duplicates()
    customer_df['join_col'] = 1
    popular_articles_cand_ver1 = customer_df.copy()
    popular_articles_cand_ver1 = popular_articles_cand_ver1.merge(MP_candidate_df, on="join_col")

    popular_articles_cand_ver1 = popular_articles_cand_ver1.drop_duplicates(subset=['profile_id','album_id'])[["profile_id","album_id"]]

    # 마지막 5주

    MP_latest_ver2_df = df_train.query(f"week == {last_week_ver2}")

    MP_df = MP_latest_ver2_df.groupby('album_id')['profile_id'].count().sort_values(ascending=False)
    MP_df = MP_df.reset_index()
    MP_df.columns = ['album_id','general_counts_ver2']
    MP_ver2 = MP_df.copy()
    MP_candidate_df = MP_df[:10].copy()
    MP_candidate_df['join_col'] = 1

    customer_df = df_train[['profile_id']].drop_duplicates()
    customer_df['join_col'] = 1
    popular_articles_cand_ver2 = customer_df.copy()
    popular_articles_cand_ver2 = popular_articles_cand_ver2.merge(MP_candidate_df, on="join_col")[["profile_id","album_id"]]

    popular_articles_cand_ver2 = popular_articles_cand_ver2.drop_duplicates(subset=['profile_id','album_id'])[["profile_id","album_id"]]

    # 마지막 6,5주 concat
    popular_articles_cand = pd.concat([popular_articles_cand_ver1, popular_articles_cand_ver2])
    popular_articles_cand = popular_articles_cand.drop_duplicates()
    # popular_articles_cand = popular_articles_cand.groupby(['profile_id','album_id'])['general_counts'].sum().reset_index()

    # general_MP_feature = popular_articles_cand[['album_id','general_counts']].drop_duplicates()

    # feature 작업
    general_mp_1 = pd.merge(MP_ver1, MP_ver2, how="left",on="album_id")
    general_mp_2 = pd.merge(MP_ver2, MP_ver1, how="left", on="album_id")
    general_MP_feature = pd.concat([general_mp_1, general_mp_2])
    general_MP_feature = general_MP_feature.drop_duplicates()

    return popular_articles_cand, general_MP_feature



def personal_most_popular(personal_train, N=5):

    personal_MP_df = personal_train.groupby(['profile_id','album_id'])[['ss_id']].count().reset_index()
    personal_MP_df.columns = ['profile_id','album_id','personal_counts']

    personal_MP_feature = personal_MP_df.copy()

    # 서로 다른날 5회 이상 시청한 앨범만
    personal_MP = personal_MP_df[personal_MP_df['personal_counts'] >= 5]
    personal_MP = personal_MP.sort_values(by=['profile_id','personal_counts'],ascending=False)

    # 한 유저가 서로 다른날 5회 이상 시청한 아이템을 나열하고 이를 count하여 MP 구함
    top_rewatch_MP_df = personal_MP.groupby("album_id").count()["personal_counts"].reset_index().sort_values(by="personal_counts", ascending=False).head(N)
    top_rewatch_MP_df = top_rewatch_MP_df.rename(columns={"personal_counts":"total_cnt"})

    # 상위 N(default=5) 개만 pick
    head_df_list = []
    # 전체 유저 대상으로 뽑기
    for user_id in tqdm(personal_train.profile_id.unique()):
        personal_MP_user_len = len(personal_MP[personal_MP['profile_id']==user_id].head(N))
        
        if personal_MP_user_len <N:
            # 상위 N개가 없는 경우
            user_df = personal_MP[personal_MP['profile_id']==user_id]
            df = pd.DataFrame()
            df['profile_id'] = [user_id for _ in range(N-len(user_df))]
            df['album_id'] = [top_rewatch_MP_df.album_id.values[num] for num in range(N-len(user_df))]
            df = pd.concat([user_df, df])
            head_df_list.append(df)
        else:
            head_df_list.append(personal_MP[personal_MP['profile_id']==user_id].head(N))

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


    # 장르 feature 생성----------------------
    user_total_watch_dict = user_genre_df.groupby('profile_id')['genre_cnt'].sum().to_dict()

    # 전체 시청 피처 만들기
    user_genre_df['user_genre_total_cnt'] = user_genre_df['profile_id'].apply(lambda x: user_total_watch_dict.get(x, None))
    # 각 유저 장르별로 퍼센트를 나타냄
    user_genre_df['user_genre_total_percent'] = user_genre_df['genre_cnt']/user_genre_df['user_genre_total_cnt']
    user_genre_feature = user_genre_df.copy()
    #---------------------------------------

    # 선호 장르 threshold 적용하여 real 선호 장르 구하기
    user_total_watch_dict = user_genre_df.groupby('profile_id')['genre_cnt'].sum()\
                            [user_genre_df.groupby('profile_id')['genre_cnt'].sum()>=34].to_dict()
   
    user_genre_df['user_genre_total_cnt'] = user_genre_df['profile_id'].apply(lambda x: user_total_watch_dict.get(x, None))
    # 각 유저 장르별로 퍼센트를 나타냄
    user_genre_df['user_genre_total_percent'] = user_genre_df['genre_cnt']/user_genre_df['user_genre_total_cnt']
    user_genre_df.drop(columns=['user_genre_total_cnt'],inplace=True)
    # na, 즉, 선호하는 장르가 없는 경우 (34회 미만 시청) 경우 전처리 됨
    user_genre_df.dropna(subset=['user_genre_total_percent'],axis=0,inplace=True)



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

    
    return genre_candidate, user_genre_feature


def age_MP(history_df, profile_df):
    age_df = pd.merge(history_df, profile_df, how="left",on="profile_id")
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
    add_proID = history_df.merge(profile_df,how="left",on="profile_id").drop(columns="album_id")
    add_proID = add_proID.merge(age_mp_cand_df,how="left",on="age")
    age_MP_candidate = add_proID[["profile_id","album_id"]]
    age_MP_candidate = age_MP_candidate.drop_duplicates()
    
    return age_MP_candidate




def series_candidate(df_train_week, threshold=30):
    series_train_week= df_train_week.copy()
    tmp = series_train_week.groupby('profile_id').log_date.max().reset_index()
    tmp.columns = ['profile_id','max_dat']
    tmp.rename(columns = {'log_date':'max_dat'},inplace=True)
    series_train_week = series_train_week.merge(tmp,on=['profile_id'],how='left')
    series_train_week['diff_dat'] = (series_train_week.max_dat - series_train_week.log_date).dt.days 
    series_train_week = series_train_week.loc[series_train_week['diff_dat']<= 6 ] 
    print('Train shape:',series_train_week.shape)

    tmp = series_train_week.groupby(['profile_id','album_id'])['log_date'].agg('count').reset_index() 
    tmp.columns = ['profile_id','album_id','ct'] 
    series_train_week = series_train_week.merge(tmp,on=['profile_id','album_id'],how='left')  
    series_train_week = series_train_week.sort_values(['ct','log_date'],ascending=False) 
    series_train_week = series_train_week.drop_duplicates(['profile_id','album_id']) 
    series_train_week = series_train_week.sort_values(['ct','log_date'],ascending=False) 

    # 1. 영상 3개 이하 본 유저 리스트 / len(cold_user) : 670
    user_list = series_train_week.profile_id.value_counts().reset_index()
    user_list.columns = ['profile_id','view_count']
    cold_user = user_list[user_list['view_count']<10].index.values.tolist()

    #판매된 아이템 개수를, 아이템 ID별로 개수 집계
    vc_ = series_train_week.album_id.value_counts().reset_index()
    vc = vc_[vc_.album_id > 10].set_index('index')
    pairs = {}
    for j,i in tqdm(enumerate(vc.index.values)):
        USERS_ = series_train_week.loc[series_train_week.album_id==i.item(),'profile_id'].unique().tolist()
        USERS = [x for x in USERS_ if x not in cold_user]
        vc2 = series_train_week.loc[(series_train_week.profile_id.isin(USERS))&(series_train_week.album_id!=i.item()),'album_id'].value_counts()
        vc2 = vc2[vc2>=threshold]
        if len(vc2.index)< 4:
            pairs[i.item()] = [vc2.index[i] for i in range(len(vc2.index))]
        else:
            pairs[i.item()] = [vc2.index[0],vc2.index[1],vc2.index[2],vc2.index[3]]

    df_list = []
    for user_id in tqdm(df_train_week.profile_id.unique()):
        df = pd.DataFrame()
        album_list = []
        for album_id in df_train_week[df_train_week["profile_id"]==user_id].album_id.unique():
            album_list += pairs.get(album_id, [])
        df["album_id"] = album_list
        df["profile_id"] = user_id
        df_list.append(df)
    series_candidate = pd.concat(df_list)
    series_candidate["album_id"] = series_candidate["album_id"].astype(int)
    series_candidate = series_candidate.drop_duplicates()

    print(series_candidate.profile_id.nunique())

    return series_candidate