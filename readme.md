Seniment Analysis on Hindi reviews.
-------------------------------------------------------------

Requirements:
1) python3 (Anaconda environment is preferred)
2) Scikit-learn
3) Numpy, Pandas
4) NLTK
5) googletrans
6) Pickle
7) Codecs


Problem:
-------------------------------------------------------------
We have used three approaches to classify the sentiment of Hindi reviews as positive or Negative.
1) Resource Based Semantic Analysis using HindiSentiWordnet.---> In this approach we used
   Hindi Sentiwordnet to classify the review's sentiment.
2) IN language Semantic Analysis. : This approach is based on training the classifiers on the same
    language as text.
3) Machine Translation Based Semantic Analysis. : In this approach we train the classifier on English
reviews and for testing, we translate the Hindi reviews into English using Googletrans api and then
we classify the Sentiment of review.

Dataset Used:
--------------------------------------------------------------
We have used a total of 1000 Hindi movie reviews for the Sentiment
Analysis. We have taken 250 labeled reviews from the dataset of IIT-
Bombay which contain 125 positive and 125 negative Hindi movie
reviews. In addition, we have manually collected 750 reviews
from a Hindi movie review website (Jagaran.com) and labeled them
as  positive  or  negative  manually.   Out  of  750  reviews  collected
manually, 375 reviews are positive and the rest 375 are negative
review.
For Machine Translation based approach, we also need english reviews, so
We have used NLTK dataset for english reviews.

Files Description:
--------------------------------------------------------------
classifiers.py : This module is used to do In-language classification. It applies
different types of classifiers on the featureset generated using Bag of word model
with feature value as TermFrequency or Term-Frequency-Inverse-Document_Frequency(TFIDF).

dbn_neuralnet.py: This module is used to do In-language classification of sentiment using
Deep belief network(DBN) as a classifier.

MachineTranslationBasedApproach.py: This module is used to do Machine Translation Based
Semantic analysis using Decision Tree as a classifier. We have used TF or TFIDF as a feature.

ResourceBasedSentimentClassification.py: This module is used to do Resource based sentiment
classification of hindi reviews using HindiSentiWordnet as a resource.

UnigramTfFeatureGeneration.py: This module is used to generate Unigram+Tf Featureset of reviews.
UnigramTfidfFeaturesetGeneration.py: This module is used to generate Unigram+Tfidf Featureset of reviews.

pos_hindi.txt: This contains positive hindi reviews of dataset. Reviews are seperated by $.
neg_hindi.txt: This contains negative Hindi reviews of dataset.
pos_english.txt: This contains positive english reviews. These are used in Machine Translation based approach.
neg_english.txt: This contains negative english reviews.

dbn_outside: This is a directory Which contain deep belief network implementation.

How To RUN:
-------------------------------------------------------------------------

1)Run on terminal 'python ResourceBasedSentimentClassification.py' to do the sentiment classification through
    HindiSentiwordnet. It is called Resource Based Semantic analysis.

2)Run on terminal 'python classifiers.py' to do In language Semantic Analysis.

3) Run on terminal 'python dbn_neuralnet. to do In-language classification through Deep Belief Networks.

3)Run on terminal 'python MachineTranslationApproach.py' to do Machine Translation Based Semantic Analysis.
