import math
import numpy as np
from tqdm import tqdm
from numpy import savetxt
from numpy import loadtxt
class Collaborative:
    def __init__(self , dataset ,num_user , num_item):
        self.dataset = dataset
        # if numbers are different with passed dataset
        self.num_user = num_user
        self.num_item = num_item
        self.dataset.pad(num_user , num_item)
        self.similarity_matrix = np.zeros((self.num_user , self.num_user))
        self.save_averages()
    def save_averages(self):
        arr = self.dataset.matrix
        mask = arr == 0
        self.user_averages = np.ma.masked_array(arr , mask).mean(axis=1)
        self.item_averages = np.ma.masked_array(arr , mask).mean(axis=0)
        self.total_average = np.average(self.dataset.matrix)
    def sim(self , u , v):
        items_in_common = []
        for j in range(self.num_item):
            if self.dataset.matrix[v,j] != 0 and self.dataset.matrix[u,j] != 0:
                items_in_common.append(j)
        if(len(items_in_common) == 0):
            return 0
        u_avg = self.user_averages[u]
        v_avg = self.user_averages[v]
        nominator = 0
        leftDenominator = 0
        rightDenominator = 0
        for i in items_in_common:
            nominator += (self.dataset.matrix[u,i] - u_avg) * (self.dataset.matrix[v,i] - v_avg)
            leftDenominator += (self.dataset.matrix[u,i] - u_avg) ** 2
            rightDenominator += (self.dataset.matrix[v,i] - v_avg) ** 2
        if rightDenominator == 0 or leftDenominator == 0:
            return 0
        return nominator / (math.sqrt(leftDenominator) * math.sqrt(rightDenominator))
    def save_all_similarities(self):
        for u in tqdm(range(self.num_user) , desc="Creating Similarity Matrix"):
            for v in range(u , self.num_user):
                self.similarity_matrix[u,v] = self.sim(u , v)
        transpose_matrix = np.transpose(self.similarity_matrix)
        for u in tqdm(range(self.num_user) , desc="Copying Similarity Matrix For Below Diagonal"):
            for v in range(u + 1 , self.num_user):
                transpose_matrix[u][v] = transpose_matrix[v][u]
        self.similarity_matrix = transpose_matrix
    def predict_per_user_average(self , u , i):
        if type(self.user_averages[u]) == np.ma.core.MaskedConstant:
            return 0
        else:
            return self.user_averages[u]
    def predict_per_item_average(self , u , i):
        if type(self.item_averages[u]) == np.ma.core.MaskedConstant:
            return 0
        else:
            return self.item_averages[u]
    def predict_global_average(self , u , i):
        return self.total_average
    # this function needs the similarity values
    # combined predict by k and predict by theta in one function
    def predict(self,u , i, k , theta):
        indices = (-self.similarity_matrix[u]).argsort()[1:k+1]
        selected = []
        for index in indices:
            if self.dataset.matrix[index , i] != 0 and self.similarity_matrix[u][index] >= theta:
                selected.append(index)
            if self.similarity_matrix[u][index] < theta:
                break
        nominator = 0
        denominator = 0
        for v in selected:
            nominator += (self.dataset.matrix[v,i] - self.user_averages[v]) * self.similarity_matrix[u , v]
            denominator += self.similarity_matrix[u , v]
        if(denominator == 0):
            return self.predict_per_user_average(u, 0)
        return self.predict_per_user_average(u, 0) + (nominator / denominator)
    def save_similarity_matrix_to_file(self , fold):
        savetxt('fold'+ str(fold) +'.csv', self.similarity_matrix , delimiter=',')
    def load_similarity_matrix_from_file(self , fold):
        self.similarity_matrix = loadtxt('fold'+ str(fold) +'.csv', delimiter=',')





