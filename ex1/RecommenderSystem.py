import time
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def read_dataset(path, size=None):
    df = pd.read_csv(path)
    df = df.dropna()
    if (size is not None):
        df = df[:size]
    return df


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


def prepare_data(df):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df.apply(lambda row: extract_sentiment(row, analyzer), axis = 1)
    df['neutral'] = np.where(df['sentiment'] != 'negative', 1, 0)
    df['positive'] = np.where(df['sentiment'] == 'positive', 1, 0)
    df['text_size'] = df.apply(lambda row: len(row['text']), axis = 1)
    return df


def train_classifier(x_train, training_labels, alph, l1_r):
    print("Train ElasticNet with alpha = " + str(alph) + " and l1_ratio = " + str(l1_r))
    model = ElasticNet(alpha=alph, l1_ratio=l1_r)
    model.fit(x_train, np.array(training_labels))
    training_predictions = model.predict(x_train)
    print("Model coefficients are: " + str(model.coef_))
    print("Model intercept is: " + str(model.intercept_))
    print('Training RMSE score: {}'.format(mean_squared_error(training_labels, training_predictions, squared=False)))
    return model


def test_classifier(classifier, x_test, testing_labels):
    print("Test ElasticNet")
    testing_predictions = classifier.predict(x_test)
    print('Testing RMSE score: {}'.format(mean_squared_error(testing_labels, testing_predictions, squared=False)))


def apply_target_encoder(x_train, x_test, y_train, column, smoothing_param):
    encoder = TargetEncoder(cols=[column], return_df=True, smoothing=smoothing_param)
    x_train = encoder.fit_transform(x_train, y_train)
    x_test = encoder.transform(x_test)
    return x_train, x_test


beginTime = time.time()
df = read_dataset('trainData.csv')
x = df[['user_id', 'text']]
y = df['stars']
x = prepare_data(x)
print("time=" + str(time.time() - beginTime))
x_train, x_test, y_train, y_test = train_test_split(x[['neutral', 'positive', 'text_size', 'user_id']], y, test_size=0.10)
x_train, x_test = apply_target_encoder(x_train, x_test, y_train, 'user_id', 1)
for alph in (0.1, 1, 10):
    for l1_r in (0.1, 0.5, 0.9):
        classifier = train_classifier(x_train, y_train, alph, l1_r)
        test_classifier(classifier, x_test, y_test)
print("time=" + str(time.time() - beginTime))
