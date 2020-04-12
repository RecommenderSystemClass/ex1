# Alex Danieli 317618718
# Gil Shamay 033076324

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from load import *


def load_trainset(path):
    df_train = load(path)
    df_train = df_train.dropna()
    df_train = df_train[['user_id', 'business_id', 'stars']]
    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(df_train, reader)
    train_set = train_data.build_full_trainset()
    return train_set


def load_testset(path):
    df_test = load(path)
    df_test = df_test.dropna()
    df_test = df_test[['user_id', 'business_id', 'stars']]
    test_set = [tuple(x) for x in df_test.to_numpy()]
    return test_set


def train_svd(train_set, K, lam, delta):
    print("Train Surprise SVD")
    model = SVD(n_factors=K, lr_all=lam, reg_all=delta)
    model.fit(train_set)
    return model


def test_svd(test_set, model):
    print("Test Surprise SVD")
    testing_predictions = model.test(test_set)
    testing_rmse_score = accuracy.rmse(testing_predictions)
    testing_mae_score = accuracy.mae(testing_predictions)
    print('Testing RMSE score: {}'.format(testing_rmse_score))
    print('Testing MAE score: {}'.format(testing_mae_score))


train_set = load_trainset('trainData.csv')
test_set = load_testset('testData.csv')
model = train_svd(train_set, 400, 0.03, 0.07)
test_svd(test_set, model)