from Classes.Collaborative import Collaborative
from Classes.Dataset import Dataset
from Classes.Evaluation import Evaluation


dataset = Dataset().read_csv('..\\dataset\\ratings.csv')
e = Evaluation(dataset)
# plotting first chart to show different algorithms
e.plot_evaluation_based_chart()
# plotting charts for parameters effects
e.plot_parameter_based_chart()