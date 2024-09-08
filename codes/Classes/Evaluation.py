import numpy as np
from .Collaborative import Collaborative
from tqdm import tqdm
import matplotlib.pyplot as plt
class Evaluation:
    def __init__(self , dataset):
        self.dataset = dataset
    def calculate_mae(self , prediction_type , k = 100000 , theta = 0):
        def func(fold , train_dataset , test_dataset):
            test_dataset.pad(self.dataset.num_user , self.dataset.num_item)
            train_dataset.pad(self.dataset.num_user , self.dataset.num_item)
            c = Collaborative(train_dataset , self.dataset.num_user , self.dataset.num_item)
            c.load_similarity_matrix_from_file(fold)
            result = 0
            count = 0
            for u in range(test_dataset.matrix.shape[0]):
                for j in range(test_dataset.matrix.shape[1]):
                    if test_dataset.matrix[u , j] != 0:
                        count += 1
                        if(prediction_type == 'predict'):
                            result += abs( c.predict(u, j, k , theta) - test_dataset.matrix[u , j])
                        if(prediction_type == 'predict_global_average'):
                            result += abs( c.predict_global_average(u, j) - test_dataset.matrix[u , j])
                        if(prediction_type == 'predict_per_item_average'):
                            result += abs( c.predict_per_item_average(u, j) - test_dataset.matrix[u , j])
                        if(prediction_type == 'predict_per_user_average'):
                            result += abs( c.predict_per_user_average(u, j) - test_dataset.matrix[u , j])
            print('fold ' + str(fold + 1) + prediction_type + ' mae is ' + str(result / count))
            return result / count
        return self.dataset.work_on_k_folds(5, func)
    def calculate_coverage(self , prediction_type , k=10000 , theta = 0):
        def func(fold , train_dataset , test_dataset):
            c = Collaborative(train_dataset , self.dataset.num_user , self.dataset.num_item)
            c.load_similarity_matrix_from_file(fold)
            nominator = 0
            count = 0
            for u in range(test_dataset.matrix.shape[0]):
                for j in range(test_dataset.matrix.shape[1]):
                    if test_dataset.matrix[u][j] != 0:
                        count += 1
                        if (prediction_type == 'predict') and c.predict(u, j, k , theta) != 0:
                            nominator += 1
                        # if (prediction_type == 'predict_by_k') and c.predict_by_k(u, j, k) != 0:
                        #     nominator += 1
                        # if (prediction_type == 'predict_by_tetha') and c.predict_by_theta(u, j, theta) != 0:
                        #     nominator += 1
                        if (prediction_type == 'predict_global_average') and c.predict_global_average(u, j) != 0:
                            nominator += 1
                        if (prediction_type == 'predict_per_item_average') and c.predict_per_item_average(u, j) != 0:
                            nominator += 1
                        if (prediction_type == 'predict_per_user_average') and c.predict_per_user_average(u, j) != 0:
                            nominator += 1
            print('fold ' + str(fold + 1) + prediction_type + ' coverage is ' + str(nominator / count))
            return nominator / count
        return self.dataset.work_on_k_folds(5, func)
    def plot_evaluation_based_chart(self):
        data = {}
        print('Calculating Evaluations Per Algorithms ...')
        data['CovUB'] = self.calculate_coverage("predict")
        print('1 / 8 passed')
        data['CovPerUserAVG'] = self.calculate_coverage("predict_per_user_average")
        print('2 / 8 passed')
        data['CovPerItemAVG'] = self.calculate_coverage("predict_per_item_average")
        print('3 / 8 passed')
        data['CovGlobAVG'] = self.calculate_coverage("predict_global_average")
        print('4 / 8 passed')
        data['MAE UB'] = self.calculate_mae("predict")
        print('5 / 8 passed')
        data['MAE PerUserAVG'] = self.calculate_mae("predict_per_user_average")
        print('6 / 8 passed')
        data['MAE PerItemAVG'] = self.calculate_mae("predict_per_item_average")
        print('7 / 8 passed')
        data['MAE GlobAVG'] = self.calculate_mae("predict_global_average")
        print('8 / 8 passed')
        print('Evaluations : ')
        print(data);
        plt.figure(figsize=(max(data.values()), 8))
        plt.bar(data.keys(), data.values())
        plt.title('Evaluation using 5-folds')
        plt.xlabel('Algorithms')
        plt.ylabel('Value')
        plt.show()
    def plot_parameter_based_chart(self):
        data1 = {}
        data2 = {}
        for theta in tqdm(range(-10 , 10) , desc='Calculating Theta Based Parameters Evaluations'):
            data1[str(theta/10)] = self.calculate_mae("predict" , k=10000 , theta=theta/10)
        for k in tqdm(range(1 , 26) , desc='Calculating K Based Parameters Evaluations'):
            data2[str(k)] = self.calculate_mae("predict" , k = k , theta=-1.0)
        data2['infinite'] = self.calculate_mae("predict" , theta=-1.0)
        print(data1)
        print(data2)
        fig, axs = plt.subplots(2, 1)
        axs[0].bar(data1.keys() , data1.values())
        axs[0].set_title('Theta')
        axs[1].bar(data2.keys(), data2.values())
        axs[1].set_title('K')
        plt.show()

            



    