from Classes.Collaborative import Collaborative
from Classes.Dataset import Dataset


dataset = Dataset().read_csv('..\\dataset\\ratings.csv')
def func(fold , train_dataset , test_dataset):
    c = Collaborative(train_dataset , train_dataset.num_user , train_dataset.num_item)
    c.save_all_similarities()
    c.save_similarity_matrix_to_file(fold)
    return 0
dataset.work_on_k_folds(5 , func)