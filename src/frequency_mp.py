import numpy as np
import pandas as pd
import pickle

df_train = pd.read_parquet("../input/lg-train-test/train_week.parquet")

# 각 유저의 영상시청 마지막 주 찾기
tmp = df_train.groupby('profile_id').log_date.max().reset_index()  
tmp.columns = ['profile_id','max_dat'] 
tmp.rename(columns = {'log_date':'max_dat'},inplace=True)
df_train = df_train.merge(tmp,on=['profile_id'],how='left') 
df_train['diff_dat'] = (df_train.max_dat - df_train.log_date).dt.days  
df_train = df_train.loc[df_train['diff_dat']<=6]

# 이전에 자주 시청한 영상 목록 확인
tmp2 = df_train.groupby(['profile_id','album_id'])['log_date'].agg('count').reset_index() 
tmp2.columns = ['profile_id','album_id','ct'] 
df_train = df_train.merge(tmp2,on=['profile_id','album_id'],how='left')  
df_train = df_train.sort_values(['ct','log_date'],ascending=False)  
df_train = df_train.drop_duplicates(['profile_id','album_id']).sort_values(['ct','log_date'],ascending=False)  

# 연계 시청 영상 목록 만들기
user_list = df_train.profile_id.value_counts().reset_index()
user_list.columns = ['profile_id','view_count']
cold_user = user_list[user_list['view_count']<3].index.values.tolist()

#판매된 아이템 개수를, 아이템 ID별로 개수 집계
vc_ = df_train.album_id.value_counts().reset_index()
vc = vc_[vc_.album_id > 3].set_index('index')
pairs = {}
for j,i in enumerate(vc.index.values):
    USERS_ = df_train.loc[df_train.album_id==i.item(),'profile_id'].unique().tolist()
    USERS = [x for x in USERS_ if x not in cold_user]
    vc2 = df_train.loc[(df_train.profile_id.isin(USERS))&(df_train.album_id!=i.item()),'album_id'].value_counts()
    pairs[i.item()] = [vc2.index[0],vc2.index[1],vc2.index[2],vc2.index[3]]
    
with open('frequency_purchased_pairs.pkl', 'wb') as f:
    pickle.dump(pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

