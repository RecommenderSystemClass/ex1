import time
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from RMSE import *
from MAE import *
from printDebug import *
from load import *


def prepare_data():
    df_train = load('./data/trainData.csv')
    df_train = df_train.dropna()
    df_test = load('./data/testData.csv')
    df_test = df_test.dropna()
    x, y_train, y_test = prepareDataForSemantic(df_train, df_test)
    return x, y_train, y_test


def prepareDataForSemantic(df_train, df_test):
    x_train = df_train[['user_id', 'text']]
    y_train = df_train['stars']
    x_test = df_test[['user_id', 'text']]
    y_test = df_test['stars']
    x = pd.concat([x_train, x_test], keys=['train', 'test'])
    return x, y_train, y_test


def split_and_reduce(x):
    x_train = x.loc['train']
    x_test = x.loc['test']
    x_train = x_train[['neutral', 'positive', 'text_size', 'user_id']]
    x_test = x_test[['neutral', 'positive', 'text_size', 'user_id']]
    return x_train, x_test


def score_to_sentiment(score):
    if score >= 0.3:
        return 'positive'
    elif score <= -0.3:
        return 'negative'
    return 'neutral'


def extract_sentiment(row, analyzer):
    all_scores = analyzer.polarity_scores(row['text'])
    score = all_scores['compound']
    return score_to_sentiment(score)


def extract_features(df):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df.apply(lambda row: extract_sentiment(row, analyzer), axis = 1)
    df['neutral'] = np.where(df['sentiment'] != 'negative', 1, 0)
    df['positive'] = np.where(df['sentiment'] == 'positive', 1, 0)
    df['text_size'] = df.apply(lambda row: len(row['text']), axis = 1)
    return df


def train_classifier(x_train, training_labels):
    print("Train LinearRegression")
    model = LinearRegression()
    model.fit(x_train, np.array(training_labels))
    training_predictions = model.predict(x_train)
    print("Model coefficients are: " + str(model.coef_))
    print("Model intercept is: " + str(model.intercept_))
    print('Training RMSE score: {}'.format(RMSE(training_labels, training_predictions)))
    print('Training MAE score: {}'.format(MAE(training_labels, training_predictions)))
    return model


def test_classifier(classifier, x_test, testing_labels):
    print("Test ElasticNet")
    testing_predictions = classifier.predict(x_test)
    print('Testing RMSE score: {}'.format(RMSE(testing_labels, testing_predictions)))
    print('Testing MAE score: {}'.format(MAE(testing_labels, testing_predictions)))


def apply_target_encoder(x_train, x_test, y_train, column, smoothing_param):
    encoder = TargetEncoder(cols=[column], return_df=True, smoothing=smoothing_param)
    x_train = encoder.fit_transform(x_train, y_train)
    x_test = encoder.transform(x_test)
    return x_train, x_test


beginTime = time.time()
x, y_train, y_test = prepare_data()
###########################
x = extract_features(x)
x_train, x_test = split_and_reduce(x)
x_train, x_test = apply_target_encoder(x_train, x_test, y_train, 'user_id', 1)
classifier = train_classifier(x_train, y_train)
test_classifier(classifier, x_test, y_test)
print("time=" + str(time.time() - beginTime))
