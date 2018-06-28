import pickle
import random

import numpy as np
from googletrans import Translator
from nltk.collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.classification import accuracy_score
from sklearn.tree import DecisionTreeClassifier as DTC

stop_words = set(stopwords.words("english"))
clf = DTC()


# This function is used to create a kind of dictionary/lexicon set
# Input to this algorithm is positive and negative reviews file which are given.
def create_lexicon(pos, neg):
    lexicon = []
    for file_name in [pos, neg]:
        with open(file_name, 'r') as f:
            contents = f.read()
            for line in contents.split('\n'):
                data = line.strip('\n')
                if data:
                    all_words = word_tokenize(data)
                    lexicon += list(map((lambda x: x.lower()), all_words))
    lexicons = []
    for word in lexicon:
        if not word in stop_words:
            lexicons.append(word)
    word_counts = Counter(lexicons)  # it will return kind of dictionary
    l2 = []
    for word in word_counts:
        if 4000 > word_counts[word]:
            l2.append(word)
    print(l2)
    return l2


def samplehandling(sample, lexicons, classification):
    featureset = []
    with open(sample, 'r', encoding="utf8") as f:
        contents = f.read()
        for line in contents.split('\n'):
            data = line.strip('\n')
            if data:
                all_words = word_tokenize(data)
                all_words = list(map((lambda x: x.lower()), all_words))
                all_words_new = []
                for word in all_words:
                    if not word in stop_words:
                        all_words_new.append(word)
                features = np.zeros(len(lexicons))
                for word in all_words_new:
                    if word in lexicons:
                        idx = lexicons.index(word)
                        features[idx] += 1
                features = list(features)
                featureset.append([features, classification])
    return featureset


def create_feature_set(pos, neg):
    featuresets = []
    lexicons = create_lexicon(pos, neg)
    featuresets += samplehandling(pos, lexicons, 1)
    featuresets += samplehandling(neg, lexicons, 0)
    random.shuffle(featuresets)
    return featuresets


def check_class(data):
    with open("decisiontree.pkl", "rb") as f:
        clf = pickle.load(f)
    prediction = clf.predict(data)
    return np.array(prediction)


def create_test_data_for_unigram(pos, neg, size):
    hm_lines = 100000
    lexicons = create_lexicon('pos_english.txt', 'neg_english.txt')
    translator = Translator()
    testset = []
    for file in [pos, neg]:
        with open(file, "r") as f:
            content = f.read()
            for line in content.split('$')[:size]:
                line = line.strip('\n')
                if not line:
                    continue
                print(line)
                line = translator.translate(line, dest="english").text
                print(line)
                featureset = np.zeros(len(lexicons))
                line = word_tokenize(line)
                words = list(set([w.lower() for w in line]))
                for w in lexicons:
                    if w in words:
                        idx = lexicons.index(w.lower())
                        featureset[idx] += 1
                featureset = list(featureset)
                if file == pos:
                    testset.append([featureset, 1])
                else:
                    testset.append([featureset, 0])
    random.shuffle(testset)
    return testset


def create_classfier_for_tf():
    featureset = create_feature_set('pos_english.txt', 'neg_english.txt')
    featureset = np.array(featureset)
    random.shuffle(featureset)
    x = list(featureset[:, 0])
    y = list(featureset[:, 1])
    print(len(x), len(y))
    clf = DTC()
    clf.fit(x, y)
    with open("decisiontree.pkl", "wb") as f:
        pickle.dump(clf, f)


def test_by_unigram():
    create_classfier_for_tf()
    testset = create_test_data_for_unigram('pos_hindi.txt', 'neg_hindi.txt', 200)
    testset = np.array(testset)
    random.shuffle(testset)
    test_x = list(testset[:, 0])
    test_y = list(testset[:, 1])
    y_pred = list(check_class(test_x))
    print(y_pred)
    print(test_y)
    print('Accuracy:  ', accuracy_score(test_y, y_pred) * 100)
    print('f-measure: ', f1_score(test_y, y_pred))


def create_classifier_for_tfidf():
    vectorizer = TfidfVectorizer(lowercase=False, analyzer=word_tokenize)
    pos = open('pos_english.txt', 'r').read()
    neg = open('neg_english.txt', 'r').read()
    documents = pos.split('\n')
    pos_count = len(documents)
    documents += neg.split('\n')
    tfidf = vectorizer.fit_transform(documents)
    featureset = tfidf.toarray()
    positive_feature_set = featureset[0:pos_count]
    negative_set = featureset[pos_count:]
    finalset = []
    for i in positive_feature_set:
        finalset.append([i, 1])
    for i in negative_set:
        finalset.append([i, 0])
    finalset = np.array(finalset)
    random.shuffle(finalset)
    train_x = list(finalset[:, 0])
    train_y = list(finalset[:, 1])
    clf = DTC()
    clf.fit(train_x, train_y)
    with open("decisiontree_tfidf.pkl", "wb") as f:
        pickle.dump(clf, f)


def translated_docs(pos, neg, size):
    translator = Translator()
    testdocuments = []
    for file in [pos, neg]:
        with open(file, "r") as f:
            content = f.read()
            for line in content.split('$')[:size]:
                line = line.strip('\n')
                if not line:
                    continue
                print(line)
                line = translator.translate(line, dest="english").text
                print(line)
                testdocuments.append(line)
    return testdocuments


def test_by_tfifdf():
    create_classifier_for_tfidf()
    vectorizer = TfidfVectorizer(lowercase=False, analyzer=word_tokenize)
    test_size = 200
    testdocuments = translated_docs('pos_hindi.txt', 'neg_hindi.txt', test_size)
    tfidf = vectorizer.fit_transform(testdocuments)
    featureset = tfidf.toarray()
    positive_feature_set = featureset[0:test_size]
    negative_set = featureset[test_size:]
    finalset = []
    for i in positive_feature_set:
        finalset.append([i, 1])
    for i in negative_set:
        finalset.append([i, 0])
    finalset = np.array(finalset)
    random.shuffle(finalset)
    test_x = list(finalset[:, 0])
    test_y = list(finalset[:, 1])
    print(test_x[1])
    with open("decisiontree_tfidf.pkl", "rb") as f:
        clf = pickle.load(f)
    y_pred = clf.predict(test_x)
    print('Accuracy:  ', accuracy_score(test_y, y_pred) * 100)
    print('f-measure: ', f1_score(test_y, y_pred))


if __name__ == '__main__':
    print("=" * 10)
    print("Unigram+Tf results")
    test_by_unigram()
    print("=" * 10)
    print("Unigram+Tfidf results")
    test_by_tfifdf()
