import pandas as pd
data = pd.read_csv("iris.data",header= None)
#data.columns=["char_a","char_b","char_c","char_d","Result"]
data_lst=[]
for i in range(150):
    temp_lst=[1]
    for j in range(4):
        temp_lst.append(data.iloc[i,j])

    data_lst.append(temp_lst)

desired = []
for i in range(150):
    if data.iloc[i,4] == 'Iris-setosa':
        desired.append([1,0,0])
    if data.iloc[i,4] == 'Iris-versicolor':
        desired.append([0,1,0])
    if data.iloc[i,4] == 'Iris-virginica':
        desired.append([0,0,1])

#print(data_lst[149])
