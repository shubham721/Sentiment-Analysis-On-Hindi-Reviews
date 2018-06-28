import codecs
import random

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = codecs.open("hindi_stopwords.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')


def get_features(type='dbn'):
    pos_reviews = codecs.open("pos_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
    neg_reviews = codecs.open("neg_hindi.txt", "r", encoding='utf-8', errors='ignore').read()
    documents = []
    # here j for adjectives, r is adverb , v is verb
    # allowed_word_types = ["J"]
    pos_count = 0
    for i in pos_reviews.split('$'):
        data = i.strip('\n')
        if data:
            documents.append(data)
            pos_count += 1
    neg_count = 0
    for i in neg_reviews.split('$'):
        data = i.strip('\n')
        if data:
            documents.append(data)
            neg_count += 1
    vectorizer = TfidfVectorizer(lowercase=False, analyzer=word_tokenize)
    tfidf = vectorizer.fit_transform(documents)
    featureset = tfidf.toarray()
    #print(vectorizer.get_feature_names())
    positive_feature_set = featureset[0:pos_count]
    negative_set = featureset[pos_count:]
    finalset = []
    for i in positive_feature_set:
        if type == 'simple':
            finalset.append([i, [1, 0]])
        else:
            finalset.append([i, 1])
    for i in negative_set:
        if type == 'simple':
            finalset.append([i, [0, 1]])
        else:
            finalset.append([i, 0])
    finalset = np.array(finalset)
    random.shuffle(finalset)
    test_size = 0.2
    testing_size = int((1 - test_size) * len(finalset))
    x_train = list(finalset[:, 0][:testing_size])  # taking features array upto testing_size
    y_train = list(finalset[:, 1][:testing_size])
    x_test = list(finalset[:, 0][testing_size:])
    y_test = list(finalset[:, 1][testing_size:])
    return x_train, y_train, x_test, y_test


get_features()
