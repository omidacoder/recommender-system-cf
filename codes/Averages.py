from Classes.Collaborative import Collaborative
from Classes.Dataset import Dataset

print('enter u :')
u = int(input())
print('enter i :')
i = int(input())

dataset = Dataset().read_csv('..\\dataset\\ratings.csv')
c = Collaborative(dataset , dataset.num_user , dataset.num_item)

print('Per User Average is : ' , c.predict_per_user_average(u, i))
print('Per Item Average is : ' , c.predict_per_item_average(u, i))
print('Global Average is : ' , c.predict_global_average(u, i))


