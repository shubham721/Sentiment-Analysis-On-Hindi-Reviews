#This function is used to fit dbn( Deep Belief network) Classifier and calculate
#  accuracy on different featuresets generated.
from statistics import mean

import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics.classification import accuracy_score

from dbn_outside.dbn.tensorflow import SupervisedDBNClassification
from UnigramTfifdFeaturesetGeneration import get_features
from UnigramTfFeatureGeneration import  create_feature_set_and_labels

def test_with_unigram_tf():
    train_x, train_y, test_x, test_y = create_feature_set_and_labels\
        ('pos_hindi.txt', 'neg_hindi.txt')
    train_x = np.array(train_x, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int32)
    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    classifier.fit(train_x, train_y)
    accuracies = []
    f_measures = []
    for i in range(1):
        y_pred = classifier.predict(test_x)
        accuracy = accuracy_score(test_y, y_pred)
        f_measure = f1_score(test_y, y_pred)
        accuracies.append(accuracy)
        f_measures.append(f_measure)
    print(accuracies)
    print('Accuracy ', mean(accuracies))
    print('F-measure', mean(f_measures))
    return


def test_with_unigram_tfidf():
    train_x, train_y, test_x, test_y = get_features('dbn')
    train_x = np.array(train_x, dtype=np.float32)
    # print(type(train_x))
    train_y = np.array(train_y, dtype=np.int32)
    test_x = np.array(test_x, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int32)
    print(type(train_x))
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    classifier.fit(train_x, train_y)
    accuracies = []
    f_measures = []
    for i in range(1):
        y_pred = classifier.predict(test_x)
        accuracy = accuracy_score(test_y, y_pred)
        f_measure = f1_score(test_y, y_pred)
        accuracies.append(accuracy)
        f_measures.append(f_measure)
    print(accuracies)
    print('Accuracy ', mean(accuracies))
    print('F-measure', mean(f_measures))
    return

if __name__ == '__main__':
    test_with_unigram_tf()
    test_with_unigram_tfidf()


