import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
path = './datasets/movielens'

class dataloader():
    def __init__(self,datapath,test_size):
        self.datapath = datapath
        self.test_size = test_size
        print("-----------------data_loading-----------------")
        ratings_df = pd.read_csv(datapath,encoding = 'utf-8')
        ratings_df.drop("timestamp", axis=1, inplace=True)
        # 유저 수
        n_unique_users = len(ratings_df['userId'].unique())
        print("유저 수:", n_unique_users)
        # 영화 수
        n_unique_movies = len(ratings_df["movieId"].unique())
        print("평가된 영화 수 :", n_unique_movies)
        print('평점 평균', ratings_df["rating"].mean())
        print('평점 표준편차', ratings_df['rating'].std())
        train_df, test_df = train_test_split(ratings_df, test_size=test_size, random_state=42)
        self.ratings_train = train_df.pivot_table('rating', index='userId', columns='movieId')
        self.ratings_train.fillna(0, inplace=True)
        #confidence
        self.user_confience = (self.ratings_train > 0).sum(axis=1)
        self.item_confidence = (self.ratings_train.T > 0).sum(axis=1)
        train_df['confidence'] = train_df.apply(
            lambda x: self.user_confience.loc[x['userId']] + self.item_confidence.loc[x['movieId']], axis=1)
        self.confidence_matrix = train_df.pivot_table(values="confidence", index='userId', columns="movieId")
        self.confidence_matrix.fillna(1, inplace=True)
        self.confidence_matrix = np.array(self.confidence_matrix)
        self.ratings_train = np.array(self.ratings_train)

        self.ratings_test = test_df.pivot_table(values="rating", index="userId", columns="movieId")
        self.ratings_test.fillna(0, inplace=True)
        self.ratings_test = np.array(self.ratings_test)
        print("-----------------data_loaded-----------------")

    def get_train_data(self):
        return self.ratings_train

    def get_test_data(self):
        return self.ratings_test

    def get_confidence(self):
        scaler = MinMaxScaler()
        minmaxconf = scaler.fit_transform(self.confidence_matrix)
        minmaxconf += 0.0001
        return minmaxconf



