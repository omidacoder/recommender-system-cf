# Author : Omid Davar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
class Dataset:
    def __init__(self):
        pass
    def read_csv(self , csv_path):
        df = pd.read_csv(csv_path , header=0)
        self.df = df
        self.num_user = df['userId'].max()
        self.num_item = df['movieId'].max()
        self.matrix = np.zeros((self.num_user , self.num_item))
        for index, row in df.iterrows():
            self.matrix[int(row['userId']) - 1 , int(row['movieId']) - 1] = row['rating'] # -1 is for starting from 0 index
        return self
    def from_dataframe(self , df):
        self.df = df
        self.num_user = df['userId'].max()
        self.num_item = df['movieId'].max()
        self.matrix = np.zeros((self.num_user , self.num_item))
        for index, row in df.iterrows():
            self.matrix[int(row['userId']) - 1 , int(row['movieId']) - 1] = row['rating'] # -1 is for starting from 0 index
        return self
    def pad(self,num_user , num_item):
        # padding the matrix
        output_shape = (num_user, num_item)
        row_pad = output_shape[0] - self.matrix.shape[0]
        col_pad = output_shape[1] - self.matrix.shape[1]
        self.matrix = np.pad(self.matrix, ((0, row_pad), (0, col_pad)), mode='constant')
    def plot_histogram(self):
        ratings = self.df['rating']
        bin_centers = np.linspace(0.5, 5, 10)
        # create the histogram
        n, bins, patches = plt.hist(ratings, bins=bin_centers, density=True, align='left')
        # calculate the bin widths
        bin_widths = bins[1:] - bins[:-1]
        percentages = n * bin_widths * 100
        # format the y-axis ticks as percentages
        plt.gca().set_yticklabels([str(x * 50) + '%' for x in plt.gca().get_yticks()])
        for i, p in enumerate(patches):
            plt.annotate(f'{percentages[i]:.1f}%', (bin_centers[i], n[i]), ha='center', va='bottom')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.title('Histogram Of User Ratings')
        plt.show()
    def get_df(self):
        return self.df
    def get_matrix(self):
        return self.matrix
    def get_k_fold(self,k):
        kf = KFold(n_splits=k)
        ratings_data = self.df
        ratings_data['fold'] = -1
        for user_id in ratings_data['userId'].unique():
            user_ratings = ratings_data[ratings_data['userId'] == user_id]
            user_ratings = user_ratings.sort_values('timestamp')
            for fold, (train_index, test_index) in enumerate(kf.split(user_ratings)):
                user_ratings.iloc[test_index, -1] = fold
            ratings_data.loc[ratings_data['userId'] == user_id, 'fold'] = user_ratings['fold']
        return ratings_data
    def work_on_k_folds(self , k , func):
        folds = self.get_k_fold(k)
        values = 0
        # Iterate over each fold
        for fold in range(k):
            # Get the training and test sets for the current fold
            train_data = folds[folds['fold'] != fold]
            val_data = folds[folds['fold'] == fold]
            values += func(fold , Dataset().from_dataframe(train_data) , Dataset().from_dataframe(val_data))
        return values / k



