import pandas as pd
from pandas import read_csv
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')

'''
异常值处理
'''
dataset = read_csv('PRSA_data_2010.1.1-2014.12.31.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
dataset['pollution'].fillna(0, inplace=True)
dataset = dataset[24:]

'''
从数据到数据集构建
'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入序列构建
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('value%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # 输出序列构建
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('value%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('value%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # 数据拼接
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 异常值处理
    if dropnan:
        agg.dropna(inplace=True)
    return agg

values = dataset.values
# 标签one hot化
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
values = values.astype('float32')
# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 数据集构建
reframed = series_to_supervised(scaled, 2, 1)
# 去除当前时刻的天气数据
reframed.drop(reframed.columns[[17,18,19,20,21,22,23]], axis=1, inplace=True)
reframed.to_csv('dataset.csv')
print(reframed.head())

'''
数据集划分，使用前三年的数据训练，其他的数据测试
'''
values = reframed.values
n_train_hours = 365 * 24 * 3
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# 数据格式 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

'''
模型构建
'''
