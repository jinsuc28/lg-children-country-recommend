# LG 유플러스 아이들나라 추천 경진대회
<img width="500" alt="메인 사진" src="https://user-images.githubusercontent.com/86936634/206629023-d19a95f7-8395-4e1b-9a0a-ab3a74b49406.png">   

- https://github.com/UpstageAI/2022-lguplus-AI-Ground

# 대회 개요

- 주제: LG유플러스의 아이들나라 데이터를 활용한 프로필별 맞춤형 콘텐츠 추천 AI 모델 개발   
- 대회기간: 2022년 11월 7일(월) 10:00 ~ 12월 2일(금) 19:00   
- 사이트: https://stages.ai/competitions/208/overview/description

# 데이터
전체 데이터 기간 : 3월~ 7월 (5개월)   
user : 8311 명 (transaction 기준)   
item : 20695 개 (transaction 기준)   
train: 1005651 row, 2개월   
test: train 속 8311명 유저가 다음 시청할 아이템@25 예측 진행,3개월   

# 대회 진행 사항
- 기존 H&M kaggle competition sota model로 lgbmrank을 기본 베이스로 대회를 진행함   
- 2 stage recommendatoin 진행 위해서 유저별로 candidate 뽑은 후    
lgbmrank 로 candidate 순위를 매겨서 추천을 진행하는 방식으로 추천을 진행
- 여러 candidate 와 feature engineering 방법을 통해서 실험 및 설계 진행함
- Notion 링크: https://jinsuc.notion.site/LG-6b9e318b7888450aa4a8852914616e16

# 1.stage Candidate Generation
# 1) candidate generation
### Most Popular @50   
- 전체 학습 기간 중 가장 많이 시청된 컨텐츠 상위 50개   
- 대부분 보는 아이템을 재추천하는 것은 가장 일반적으로 사용하는 candidate 이며 데이터 분석 결과    
중복 추천이 큰 의미를 가지는 task라고 판단하여 상위 50개 정도를 candidate 선정

### General Most Popular @20
- 마지막 6주 MP 10개, 마지막 7주 MP 10개    
- 추천은 다음번 유저가 시청할 아이템을 추천해야는 만큼 마지막에 해당하는    
아이템일수록 높은 candidate 후보에 넣어야 한다고 판단함.

### Personal Most Popular @50
- 유저별 MP50 추천 진행   
- 데이터 분석 결과 유저 개인이 시청했던 아이템을 거의 무조건 재시청하는 것으로 파악함   
따라서 유저별로 많이 시청한 아이템을 재 추천해주기 위한 candidate 진행

### User Genre Most Popular @20
- 유저가 선호하는 아이템 장르가 있을 것이라고 판단하였으며    
각 장르별로 MP를 구하였으며 유저별 선호하는 top2 장르별 10개씩 candiate 선정

### ALS MF(중복 포함) @100
- 중복 시청 제거 없이 ALS 학습을 진행하였으며 각 유저별 100개 candidate   
- H&M discussion 통해 다른 모델을 통한 candidate 하면 성능이 올라가는 것을 확인하여   
진행하였으며 성능적으로도 유의미한 cadidate가 많은 것으로 판단함

### Apriori @5
- 같이 많이 시청된 아이템을 candidate 하기 위해 효과적인 candidate 방법이라고 판단하여 진행   
- support 최소 0.1 이상, min confidence 0.8로 설정하였으며 lift 로 정렬하여서    
유저가 해당 아이템을 시청 이력이 있다면 candidate 담는 방식으로 진행    
결론적으로 apriori@5는 다른 candidate 포함되는 경우가 많아 효과가 없는 것으로 판단되어 제외함

### Nueral CF @25
- base model NCF 통해 candidate 25를 뽑아 진행
- ALS와 같은 이유로 진행

### AZPS(AmaZon Personalize Sagemaker) @100
- 아마존 퍼스널라이저를 통해 튜닝후 candidate@100 선정
- transaction 없는 아이템이 있었으며 candidate 아이템들이 다른 candidate 와는 다르게   
feature 생성할 수 없어서 성능이 많이 하락하는 것을 확인하여 결론적으로 제외함

### Series candidate @25
- 아이템의 시리즈가 있을 것으로 판단하였으며 각 아이템별로 연속적으로 혹은 같이 시청한 아이템을    
모두 count 하여 threshold를 정하여 candidate@25 뽑아 진행   
- 유저가 연속 시청 목록을 많이 시청하는 것으로 판단하였으며 실제 연속 시청 아이템들은    
시리즈 아이템이 많은 것으로 판단함

### Age candidate @N
- 나이별로 많이 시청되는 아이템이 있을 것이라고 판단하여 candidate 진행   
하지만, 성능이 좋지 않아 제외함

### 결론
- 총 유저별 약 @400 candidate 추출하였으며 최종적으로 Apriori, AZPS, Age cadidate 제외한    
총 7가지 candidate 방법을 통해 추천을 진행


# 2) feature engineering

### ALS User, item vector
- 각 vector factor 200 뽑아 진행하였으며 다양한 아이템 및 유저의 interation vector 통해   
유의미한 성능 향상 확인

### day, week
- 유저와 아이템의 시청 시간을 넣어주기 위해서 유저와 아이템에서 transaction 있을 경우   
feauture 넣어주게 됨. 없는 경우 nan

### MP count feature
- 아이템 별로 전체 몇번 시청되었는지 카운트를 하였으며 이를 feature 넣어줌   
- 아이템 별로 많이 시청되는 아이템인지 아닌지를 판단할 수 있는 피처라고 판단함(많이 시청된 아이템이 높은 rank 가지도록)

### General MP count feature
- 각 마지막 두 주별로 몇번 시청되었는지 아이템별로 count 진행하였여 두개의 feature 만들어 진행

### Personal count feautre
- 각 유저별로 많이 시청한 아이템을 conut 하여 각 유저별로 피처를 만듬
- 각 유저별 선호 아이템에 대한 가중치 부여, 가장 유의미한 피처로 판단 됨

### meta feature
- meta 데이터 feature 중 사용할 수 있는 피처를 category화 하여 사용

### genre user count feautre
- 각 유저별로 선호하는 장르 두개를 count 하여 feauture 사용

### watch continuos binary feature
- 연속 시청한 적이 있는 아이템은 1 아니면 0 부여하여 사용

### profile feature
- 나이, 성별 및 사용할 수 있는 feature categorycal 하여 사용

### favorite cast feature
- 각 유저별로 선호하는 cast 하나를 선정하여 categorycal feature로 사용

### favorite keyword feature
- 각 유저별로 선호하는 keyword 선정하여 categorycal feature 사용

# 2.LGBMRank   
# HyperParameter   
- num leaves :15   
- learning rate: 0.005   
- n-estimators: 35

## label
- last 1 week
- 마지막 한주의 유저와 아이템의 interaction 있었으면 candidate pos로 부여하는 식으로 사용

# 좋았던 점
- 빠르게 공통적으로 진행하는 사항은 pipline화 한 것
- 빠른 문제 정의 및 실험 대상 train split
- 노션을 통한 실험 결과 공유

# 아쉬운 점
- Pure ALS MF 를 중복 포함으로 학습한 결과 public 21위(전체 216팀), private 26위 성적이였지만    
lgbmrank 모델 사용으로 인해 많은 순위 하락 서로 다른 성격에 모델을 제출했어야 한다고 판단됨
- 좋은 ALS MF 모델부터 build-up 하지 못한 것이 아쉬움
- 다양한 model에 대한 실험 진행
- test 가 마지막 3달을 예측하는 것인 만큼 label도 긴 기간으로 설정하여야    
되거나 다른 방법을 설정해야한 것은 아닌지 생각됨