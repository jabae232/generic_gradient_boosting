import csv
import numpy as np
from numpy.core import numeric
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def convertData(arrayOfArguments, data, position):
    # replaces a string argument with a float
    for i in data:
        for idj, j in enumerate(arrayOfArguments):
            if i[position] == j:
                i[position] = float(idj)
    return data


def convertDataToFloat(data):
    obj = []
    print(data.shape)
    for i in range(5):
        print("index", i)
        x = data[:, i].astype(np.float64)
        obj.append(x)
    return obj


# read data
data = list(csv.reader(open('./weatherHistory.csv')))
data = np.asarray(data)

# delete unnecessary columns
data = np.delete(data, (0), axis=0)
data = np.delete(data, (11), axis=1)
data = np.delete(data, (10), axis=1)
data = np.delete(data, (9), axis=1)
data = np.delete(data, (2), axis=1)
data = np.delete(data, (1), axis=1)
data = np.delete(data, (0), axis=1)
# temperature
y_data = data[:, 2]
data = np.delete(data, (2), axis=1)
x_data = data


# these 3 is to get the nonduplicate array of string arguments
daily_summary = list(set(x_data[:, -1]))
summary = list(set(x_data[:, 0]))
precip_type = list(set(x_data[:, 1]))

x_data = convertDataToFloat(x_data)
obj = []
x = y_data.astype(np.float64)
obj.append(x)
y_data = obj
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

print(x_data.shape, y_data.shape)
x_data = np.reshape(x_data, (96453, 5))
y_data = np.reshape(y_data, (96453, ))
print(x_data.shape, y_data.shape)

# split and convert to np
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


DT_regressor = DecisionTreeRegressor()
DT_regressor.fit(x_train, y_train)
prediction = DT_regressor.predict(x_test)
mean_squared_score = mean_squared_error(y_test, prediction)
print("decision tree mean squared error: ", mean_squared_score)

# GB_regressor = GradientBoostingRegressor(n_estimators=200)
# GB_regressor.fit(x_train, y_train)
# accuracy = GB_regressor.score(x_test, y_test)
# print("gradient boosting accuracy: ", accuracy)
