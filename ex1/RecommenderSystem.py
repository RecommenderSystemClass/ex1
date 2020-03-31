import time
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import pandas as pd

from gensim.models import doc2vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def read_dataset(path, size):
    dataset = pd.read_csv(path)
    dataset = dataset[:size]
    x_train, x_test, y_train, y_test = train_test_split(dataset.text, dataset.stars, random_state=0, test_size=0.30)
    data = x_train.tolist() + x_test.tolist()
    x_train = label_sentences(x_train, 'Train')
    x_test = label_sentences(x_test, 'Test')
    all = label_sentences(data, 'All')
    return x_train, x_test, y_train, y_test, all


def clean_text(text):
    tokenized_text = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    no_stop_words_tokenized_text = [w for w in tokenized_text if not w in stop_words]
    porter = PorterStemmer()
    no_stop_words_stemmed_tokenized = [porter.stem(w) for w in no_stop_words_tokenized_text]
    return no_stop_words_stemmed_tokenized


def label_sentences(corpus, label_type):
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(clean_text(v), [label]))
    return labeled


def get_vectors(doc2vec_model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        index = i
        if vectors_type == 'Test':
            index = index + len(x_train)
        prefix = 'All_' + str(index)
        vectors[i] = doc2vec_model.docvecs[prefix]
    return vectors


def train_doc2vec(corpus):
    print("Building Doc2Vec model")
    d2v = doc2vec.Doc2Vec(min_count=1, window=3, vector_size=100, sample=1e-3, seed=1, workers=5)
    d2v.build_vocab(corpus)
    return d2v


def train_classifier(d2v, training_vectors, training_labels):
    print("Train Doc2Vec on training set")
    d2v.train(training_vectors, total_examples=len(training_vectors), epochs=d2v.iter)
    train_vectors = get_vectors(d2v, len(training_vectors), 100, 'Train')
    model = RandomForestClassifier(n_jobs=4, verbose=3)
    model.fit(train_vectors, np.array(training_labels))
    training_predictions = model.predict(train_vectors)
    print('Training predicted classes: {}'.format(np.unique(training_predictions)))
    print('Training accuracy: {}'.format(accuracy_score(training_labels, training_predictions)))
    print('Training F1 score: {}'.format(f1_score(training_labels, training_predictions, average='weighted')))
    return model


def test_classifier(d2v, classifier, testing_vectors, testing_labels):
    print("Train Doc2Vec on testing set")
    d2v.train(testing_vectors, total_examples=len(testing_vectors), epochs=d2v.iter)
    test_vectors = get_vectors(d2v, len(testing_vectors), 100, 'Test')
    testing_predictions = classifier.predict(test_vectors)
    hist1, bins1 = np.histogram(testing_predictions, bins=np.linspace(1, 5, 6))
    print('Predictions: ' + str(bins1) + ' are ' + str(hist1))
    hist2, bins2 = np.histogram(testing_labels, bins=np.linspace(1, 5, 6))
    print('Labels: ' + str(bins2) + ' are ' + str(hist2))
    print('Testing predicted classes: {}'.format(np.unique(testing_predictions)))
    print('Testing accuracy: {}'.format(accuracy_score(testing_labels, testing_predictions)))
    print('Testing F1 score: {}'.format(f1_score(testing_labels, testing_predictions, average='weighted')))


beginTime = time.time()
x_train, x_test, y_train, y_test, all = read_dataset('trainData.csv', 500000)
d2v_model = train_doc2vec(all)
classifier = train_classifier(d2v_model, x_train, y_train)
test_classifier(d2v_model, classifier, x_test, y_test)
took = time.time() - beginTime
print("time=" + str(took))