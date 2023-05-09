import numpy as np
import json
import sys
import nltk
import logging
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter
from os import add_dll_directory, remove
from nltk.stem import WordNetLemmatizer, wordnet
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from itertools import chain, starmap
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models, matutils
from gensim.models import Word2Vec
# nltk.download()
# import local modules
import DIIM_config as config


# create logger
logger = logging.getLogger(config.LOGAPPLICATION_NAME)


class Auxiliary:
    def __init__(self):
        self.logger = logging.getLogger(config.LOGAPPLICATION_NAME)
        self.logger.info('creating an instance of AnalyseJSONData')

    def do_something(self):
        self.logger.info('doing something')
        a = 1 + 1
        self.logger.info('done doing something')


def get_synonyms(keys_list):
    synsets = []
    for key in keys_list:
        synsets = wordnet.synsets(key)
        if (len(synsets) > 0):
            logger.info("Synonyms of :" + str(key) + ' are : ' + str(synsets))
    return synsets


def get_lemmas(synsets):

    dict_lemmas = {}
    for syn in synsets:
        # logger.info(str(syn))
        lemmas = syn.lemmas()
        if len(lemmas) == 0:
            logger.info("No lemmas found for " + str(syn))
        else:
            dict_lemmas[syn] = lemmas
        logger.info(" Number of found lemmas for " + str(syn) + ' are : ' + str(len(lemmas)))

    return dict_lemmas


def doStemming(tokens):
    # Stemming
    porter = PorterStemmer()
    stem_words = np.vectorize(porter.stem)
    stemed_text = ' '.join(stem_words(tokens))
    logger.info(f"nltk stemed text: {stemed_text}")


def extractTermsWithTF_IDF(cleanDatasetFilename):

    with open(cleanDatasetFilename) as f:
        contents = f.read()
    corpus = nltk.sent_tokenize(contents)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    logger.info(vectorizer.get_feature_names_out())
    # print(X.shape)
    tf_idf_model = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
    # save results in a text file
    tFIDFDatasetFilename = cleanDatasetFilename + "TFIDF.txt"
    fileTFIDF = open(tFIDFDatasetFilename, 'w')
    for key, val in tf_idf_model.items():
        # print (str(key) + ':' + str(val))
        fileTFIDF.write(str(key) + ':' + str(val) + '\n')
    fileTFIDF.close()
    # sort the dictionary of words by tf-idf and save the results in a text file
    fileTFIDF_Sorted = open('WikiDatasetTFIDF_Sorted.txt', 'w')
    listofwords = sorted(tf_idf_model.items(), reverse=True, key=lambda x: x[1])
    for elem in listofwords:
        logger.info(elem[0], " ::", elem[1])
        fileTFIDF_Sorted.write(str(elem[0]) + ':' + str(elem[1]) + '\n')
    fileTFIDF_Sorted.close()
