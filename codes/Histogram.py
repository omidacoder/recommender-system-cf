from Classes.Dataset import Dataset

# ploting histogram
dataset = Dataset().read_csv('..\\dataset\\ratings.csv')
dataset.plot_histogram()
