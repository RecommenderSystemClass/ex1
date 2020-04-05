import time
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def read_dataset(path, size):
    dataset = pd.read_csv(path)
    dataset = dataset.dropna()
    dataset = dataset[:size]
    x_train, x_test, y_train, y_test = train_test_split(dataset.text, dataset.stars, test_size=0.10)
    return x_train, x_test, y_train, y_test


def create_bow(x_train, x_test):
    stemmer = PorterStemmer()
    analyzer = CountVectorizer().build_analyzer()

    def stemmed_words(doc):
        return (stemmer.stem(w) for w in analyzer(doc))

    stem_vectorizer = CountVectorizer(analyzer=stemmed_words, max_df=0.8, max_features=1000)
    all = pd.concat([x_train, x_test])
    bow = stem_vectorizer.fit_transform(all).toarray()
    size_of_train = len(x_train)
    vectorized_x_train = bow[:size_of_train, :]
    vectorized_x_test = bow[size_of_train:, :]
    return vectorized_x_train, vectorized_x_test


def train_classifier(x_train, training_labels):
    print("Train Doc2Vec on training set")
    model = RandomForestClassifier(n_jobs=4, verbose=3)
    model.fit(x_train, np.array(training_labels))
    training_predictions = model.predict(x_train)
    print('Training predicted classes: {}'.format(np.unique(training_predictions)))
    print('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    print('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model


def test_classifier(classifier, x_test, testing_labels):
    print("Train Doc2Vec on testing set")
    testing_predictions = classifier.predict(x_test)
    hist1, bins1 = np.histogram(testing_predictions, bins=np.linspace(1, 5, 6))
    print('Predictions: ' + str(bins1) + ' are ' + str(hist1))
    hist2, bins2 = np.histogram(testing_labels, bins=np.linspace(1, 5, 6))
    print('Labels: ' + str(bins2) + ' are ' + str(hist2))
    print('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    print('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    print('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))


beginTime = time.time()
x_train, x_test, y_train, y_test = read_dataset('trainData.csv', 200000)
vectorized_x_train, vectorized_x_test = create_bow(x_train, x_test)
classifier = train_classifier(vectorized_x_train, y_train)
test_classifier(classifier, vectorized_x_test, y_test)
took = time.time() - beginTime
print("time=" + str(took))
