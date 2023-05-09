import logging
from pprint import pprint
import gensim
import pickle
import os.path as path
import os
from gensim.models import LsiModel
from gensim import models
from gensim import corpora
from collections import defaultdict
import pandas as pd

# local imports
import DIIM_Corpora_Onto as corpora_onto
import DIIM_config as config

logger = config.logger

DIIM_Ontos_LSI_Models = []
DIIM_Onto_Dictionaries = []


def get_DIIM_Ontos_LSI_Models():
    if len(DIIM_Ontos_LSI_Models) == 0:
        load_DIIM_Onto_LSI_Models()
    return DIIM_Ontos_LSI_Models


def load_DIIM_Onto_LSI_Models():

    DIIM_Ontos_Corpora = corpora_onto.get_DIIM_Ontos_Corpora()
    Ontology_Names = corpora_onto.get_Ontology_Names()

    index = 0
    for corpus in DIIM_Ontos_Corpora:
        ontology_name = Ontology_Names[index]
        index = index + 1
        load_Onto_LSI_model(ontology_name, corpus)


def load_Onto_LSI_model(ontology_name, documents):
    lsi_path = os.path.join(config.DIIM_LSI_ONTO_STORE_PATH,  ontology_name)

    if (os.path.exists(lsi_path)):
        logger.info('Loading model from  ' + lsi_path)
        lsi_model = LsiModel.load(lsi_path)  # load model
        logger.info(' Loading of saved ' + ontology_name + ' model done!')
    else:
        logger.info('Createing new LSI model for  ' + ontology_name)
        lsi_model = build_Onto_LSI_model(ontology_name, lsi_path, documents)

    DIIM_Ontos_LSI_Models.append(lsi_model)


def build_Onto_LSI_model(ontology_name, lsi_path, documents):
 # remove words that appear only once
    frequency = defaultdict(int)
    # new_corpus = w2v_utils.drop_empty_rowsandcolums(corpus)
    #new_corpus = corpus.replace(to_replace='None', value=np.nan).dropna()
    none_counter = 0
    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents]

    # print(texts[0])
    # print(texts[1])
    # print(texts[2])
    # print(texts[3])
    # print('No of nones', none_counter)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts]

    dictionary = corpora.Dictionary(texts)
    DIIM_Onto_Dictionaries.append(dictionary)
    # print(dictionary)

    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]

    # initialize an LSI transformation
    lsi_model = models.LsiModel(
        corpus_tfidf, id2word=dictionary, num_topics=2)
    # create a double wrapper over the original corpus: bow->tfi
    corpus_lsi = lsi_model[corpus_tfidf]
    lsi_model.save(lsi_path)

    return lsi_model


def print_topics():
    index = 0
    print('Top 10 topics for ontology models ')
    ontology_Names = corpora_onto.get_Ontology_Names()
    for lsi_model in DIIM_Ontos_LSI_Models:
        print('\nTop 10 topics from ' + ontology_Names[index] + ':')
        print(type(lsi_model))
        # print(lsi_model.get_topics())
        pprint(lsi_model.print_topics())
        index = index + 1


def print_dictionaries():
    for dictionary in DIIM_Onto_Dictionaries:
        print(dictionary)
        print(dictionary.id2token)
        print('*********')


def initialize():
    load_DIIM_Onto_LSI_Models()
    print_topics()
    print_dictionaries()


initialize()
