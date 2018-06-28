import random
from collections import Counter

import numpy as np
from googletrans import Translator
from nltk.tokenize import word_tokenize
import codecs
from dbn_outside.dbn.tensorflow import SupervisedDBNClassification

hm_lines = 5000000
translator = Translator()
stopwords = codecs.open("hindi_stopwords.txt", "r", encoding='utf-8', errors='ignore').read().split('\n')


# Creating a set of lexicons which is a kind of dictionary of words.
def create_lexicon(pos, neg):
    lexicon = []
    for file_name in [pos, neg]:
        with codecs.open(file_name, 'r',encoding='utf-8',errors='ignore') as f:
            contents = f.read()
            for line in contents.split('$'):
                data = line.strip('\n')
                if data:
                    all_words = word_tokenize(data)
                    lexicon += list(all_words)
    lexicons = []
    for word in lexicon:
        if not word in stopwords:
            lexicons.append(word)
    word_counts = Counter(lexicons)  # it will return kind of dictionary
    l2 = []
    for word in word_counts:
        if 60 > word_counts[word]:
            l2.append(word)
    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []
    with codecs.open(sample, 'r', encoding="utf8",errors='ignore') as f:
        contents = f.read()
        for line in contents.split('$'):
            data = line.strip('\n')
            if data:
                all_words = word_tokenize(data)
                all_words_new = []
                for word in all_words:
                    if not word in stopwords:
                        all_words_new.append(word)
                features = np.zeros(len(lexicon))
                for word in all_words_new:
                    if word in lexicon:
                        idx = lexicon.index(word)
                        features[idx] = 1
                features = list(features)
                featureset.append([features, classification])
    return featureset


def create_feature_set_and_labels(pos, neg, test_size=0.2):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, 1)
    features += sample_handling(neg, lexicon, 0)
    random.shuffle(features)
    features = np.array(features)
    #print(len(features))
    testing_size = int((1 - test_size) * len(features))

    x_train = list(features[:, 0][:testing_size])  # taking features array upto testing_size
    y_train = list(features[:, 1][:testing_size])  # taking labels upto testing_size

    x_test = list(features[:, 0][testing_size:])
    y_test = list(features[:, 1][testing_size:])
    return x_train, y_train, x_test, y_test


def check_class(text, lexicon):
    line = translator.translate(text, dest='hi').text
    classifier = SupervisedDBNClassification.load('dbn.pkl')
    predict_set = []
    all_words = word_tokenize(line)
    # all_words = [lemmatizer.lemmatize(i) for i in all_words]
    features = np.zeros(len(lexicon))
    for word in all_words:
        if word in lexicon:
            idx = lexicon.index(word)
            features[idx] += 1
    features = list(features)
    predict_set.append(features)
    predict_set = np.array(predict_set, dtype=np.float32)
    predict_set = classifier.predict(predict_set)
    #print(predict_set)


def create_feature_set_and_labels_simple(pos, neg, test_size=0.2):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)
    features = np.array(features)
    #print(len(features))
    testing_size = int((1 - test_size) * len(features))

    x_train = list(features[:, 0][:testing_size])  # taking features array upto testing_size
    y_train = list(features[:, 1][:testing_size])  # taking labels upto testing_size

    x_test = list(features[:, 0][testing_size:])
    y_test = list(features[:, 1][testing_size:])
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    create_lexicon('pos_hindi.txt', 'neg_hindi.txt')
    # x_train,y_train,x_test,y_test = create_feature_set_and_labels('pos_hindi.txt','neg_hindi.txt')
    # print (x_train[0])
    # with open('sentiment_data.pickle','wb') as f:
    #   pickle.dump([x_train,y_train,x_test,y_test],f)
    # lexicon = create_lexicon('pos_hindi.txt','neg_hindi.txt')
    # check_class('while the performances are often engaging , this loose collection of largely improvised numbers would probably have worked better as a one-hour tv documentary . \
    # interesting , but not compelling . ',lexicon)
