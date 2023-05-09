###########################
# TODO: update documenation
# Docs: https://www.kaggle.com/code/jeffd23/visualizing-word-vectors-with-t-sne/notebook
#
# This script has following main routines
# 1) Ontology Dir ---convert & ---> Corpora raw: Words in ontologies as Dataframe (store them as pkl)
# 2) Dataframe Corpora raw ---preprocess---> Cleaned Dataframe Corpora
# 3) Cleaned Dataframe Corpora ---convert---> Word2Vec models of ontologies
# 4) Display scatter plots of Word2Vec of ontologies
##########################

# python imports

from gensim.models import LsiModel
from gensim import models
from gensim.parsing.preprocessing import preprocess_documents
from gensim import corpora
from collections import defaultdict
import pickle
import os
from pathlib import Path
# import string
# import re

# # matplot imports
# import matplotlib.pyplot as plt
# from matplotlib import pyplot
# # numpy imports
# import numpy as np
# pandas imports
import pandas as pd
# # sklearn imports
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# gensim imports
# from gensim.models import word2vec
# from gensim.test.utils import common_texts
# from gensim.models import Word2Vec
# from gensim.parsing.preprocessing import preprocess_documents
# from gensim import utils
# from gensim.parsing.porter import PorterStemmer
# from gensim.parsing import stem_text
# nltk imports
import nltk
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
# misc imports
from traceback import print_tb
from string import ascii_lowercase

# local imports
import DIIM_config as config
import DIIM_word2vec_utils as w2v_utils

# variables
logger = config.logger


# pre-processed tokens/words created while reading files in the IoT dir
DIIM_IoT_DF_Corpora_raw = []  # list of dataframes

# pre-processed tokens in rows and columns
# removes empty rows
DIIM_IoT_DF_Corpora = []  # list of dataframes

# names of the IoT (directory names used to label an IoT data)
IoT_Names = []  # list of strings

# Test word from the IoT data on water quality monitoring
IoT_TEST_WORDS = ['sensor', 'system', 'time', 'measurement', 'digital', 'platform',
                  'actuator', 'data', 'value', 'property', 'determinand',
                  'process', 'location', 'position', 'temperature', 'description', 'pH', 'Clorine']  # , 'temp'

DIIM_IoT_Corpora = []

DIIM_IoT_LSI_Models = []


def get_index_of_iot(iot_dir_name):
    if len(IoT_Names) == 0:
        iot_Names = get_IoT_Names()

    if iot_dir_name in IoT_Names:
        return IoT_Names.index(iot_dir_name)
    else:
        return None


def get_IoT_Names():
    if len(IoT_Names) == 0:
        for root, dirs, files in os.walk(config.IOT_DIR):
            for dir in dirs:
                # add the ontolgy name to the list
                IoT_Names.append(dir)

    return IoT_Names


def get_DIIM_IoT_DF_Corpora_raw():
    if len(DIIM_IoT_DF_Corpora_raw) == 0:
        load_DIIM_IoT_DF_Corpus_raw()

    return DIIM_IoT_DF_Corpora_raw


def get_DIIM_IoT_DF_Corpora():
    if len(DIIM_IoT_DF_Corpora) == 0:
        load_DIIM_IoT_DF_Corpus()

    return DIIM_IoT_DF_Corpora


def get_DIIM_IoT_Corpora():
    if len(DIIM_IoT_Corpora) == 0:
        load_DIIM_IoT_Corpora()

    return DIIM_IoT_Corpora


def build_DIIM_IoT_DF_Corpus_raw(root, dir):
    DIIM_IoT_Dir_path = Path(os.path.join(root, dir))
    DIIM_IoT_DF_Corpus_raw_path = Path(config.DIIM_DF_STORE_PATH, dir)
    DIIM_IoT_DF_Corpus_raw_path = str(
        DIIM_IoT_DF_Corpus_raw_path) + "_IoT_DF_Corpus_raw.pkl"
    DIIM_IoT_DF_Corpus_raw_path_CSV = str(
        DIIM_IoT_DF_Corpus_raw_path) + "_IoT_DF_Corpus_raw.csv"

    logger.info('Trying to load stored DIIM IoT Dataframe corpus raw from ' +
                DIIM_IoT_DF_Corpus_raw_path)
    # df_corpus_raw = []
    # if a raw corpus of the IoT alread read and created load it
    if (os.path.exists(DIIM_IoT_DF_Corpus_raw_path)):
        df_corpus_raw = pd.read_pickle(DIIM_IoT_DF_Corpus_raw_path)
        logger.info(' loading of DIIM IoT Corpus raw DATA done!')
    else:
        # else create a raw corpus of the ontology and save it
        logger.info(' file not found!')
        logger.info('Building Dataframe corpus raw from ',
                    DIIM_IoT_DF_Corpus_raw_path)
        df_corpus_raw = w2v_utils.build_corpora_from_DIR(DIIM_IoT_Dir_path)

        logger.info(' saving newly created DF Corpus raw to ',
                    DIIM_IoT_DF_Corpus_raw_path)
        df_corpus_raw.to_pickle(DIIM_IoT_DF_Corpus_raw_path)
        df_corpus_raw.to_csv(DIIM_IoT_DF_Corpus_raw_path_CSV)

    # print(DIIM_Ontology_Dir_path)
    # print(df_corpus_raw.head())
    # print(df_corpus_raw.shape)
    return df_corpus_raw


def build_DIIM_IoT_DF_Corpus(root, iot_dir):
    DIIM_IoT_Dir_path = Path(os.path.join(root, iot_dir))
    DIIM_IoT_DF_Corpus_path = Path(config.DIIM_DF_STORE_PATH, iot_dir)
    DIIM_IoT_DF_Corpus_path = str(
        DIIM_IoT_DF_Corpus_path) + "_IoT_DF_Corpus.pkl"
    DIIM_IoT_DF_Corpus_path_CSV = str(
        DIIM_IoT_DF_Corpus_path) + "_IoT_DF_Corpus.csv"

    logger.info('Trying to load stored DIIM IoT Dataframe corpus from ' +
                str(DIIM_IoT_DF_Corpus_path))
    df_corpus = []

    # if a raw corpus of the ontology alread read and created load it
    if (os.path.exists(DIIM_IoT_DF_Corpus_path)):
        df_corpus = pd.read_pickle(DIIM_IoT_DF_Corpus_path)
    else:
        # else create a raw corpus of the ontology and save it
        logger.info(' file not found!')
        logger.info('Building IoT Dataframe corpus raw from ' +
                    DIIM_IoT_DF_Corpus_path)

        # Assure that list of DIIM_DataFrame_Corpora_raw has elements
        # and load_DIIM_DataFrame_Corpora_raw is executed
        if len(DIIM_IoT_DF_Corpora_raw) <= 0:
            load_DIIM_IoT_DF_Corpus_raw()

        # get the dataframe from the index of ontology name
        df_corpus = DIIM_IoT_DF_Corpora_raw[get_index_of_iot(iot_dir)]

        # clean the dataframe
        # usefull for sensor data in json format,
        # where key value is repeated for every measurement
        # df_corpus = df_corpus.drop_duplicates() # it is not good for IoT!?
        df_corpus = w2v_utils.drop_empty_rowsandcolums(df_corpus)

        logger.info(' saving cleaned IoT DF Corpus to ' +
                    str(DIIM_IoT_DF_Corpus_path))
        df_corpus.to_pickle(DIIM_IoT_DF_Corpus_path)
        df_corpus.to_csv(DIIM_IoT_DF_Corpus_path_CSV)

    logger.info(' loading of IoT Dataframe corpus done!')

    return df_corpus


def build_DIIM_IoT_DF_Corpora_raw(iot_dir):
    logger.info("Initializing DIIM IoT DataFrame corpus raw from " + iot_dir)
    # ontologies_dict_counter = 0  # counter for ontologies
    # number_of_read_files = 0
    for root, dirs, files in os.walk(iot_dir):
        for dir in dirs:
            # add the ontolgy name to the list
            IoT_Names.append(dir)
            diim_iot_df_corpus_raw = build_DIIM_IoT_DF_Corpus_raw(root, dir)
            # print(diim_iot_df_corpus_raw.head())
            print('Shape of raw Dataframe ', dir,
                  ': ', diim_iot_df_corpus_raw.shape)
            DIIM_IoT_DF_Corpora_raw.append(diim_iot_df_corpus_raw)


def build_DIIM_IoT_DF_Corpora(iot_dir):

    logger.info("Initializing DIIM IoT DataFrame corpus from " + iot_dir)
    # ontologies_dict_counter = 0  # counter for ontologies
    # number_of_read_files = 0
    for root, dirs, files in os.walk(iot_dir):
        for dir in dirs:
            # donot add the ontolgy name to the list because it was
            # already added when raw corpus was created
            # IoT_Names.append(dir)
            diim_iot_df_corpus = build_DIIM_IoT_DF_Corpus(root, dir)
            # print(diim_iot_df_corpus.head())
            print('Shape of Dataframe ', dir,
                  ': ', diim_iot_df_corpus.shape)
            DIIM_IoT_DF_Corpora.append(diim_iot_df_corpus)


def load_DIIM_IoT_DF_Corpus():
    print('Loading IoT corpora list')
    print('Length of IoT corpora list: ',
          len(DIIM_IoT_DF_Corpora))

    if len(DIIM_IoT_DF_Corpora) <= 0:
        build_DIIM_IoT_DF_Corpora(config.IOT_DIR)

    print('Length of IoT DF corpora list after loading/building: ',
          len(DIIM_IoT_DF_Corpora))


def load_DIIM_IoT_DF_Corpus_raw():
    print('Loading IoT corpora raw list')
    print('Length of IoT corpora raw list: ',
          len(DIIM_IoT_DF_Corpora_raw))

    if len(DIIM_IoT_DF_Corpora_raw) == 0:
        build_DIIM_IoT_DF_Corpora_raw(config.IOT_DIR)

    print('Length of IoT corpora raw list after loading/building: ',
          len(DIIM_IoT_DF_Corpora_raw))


def load_DIIM_IoT_Corpora():

    diim_IoT_DF_Corpora = get_DIIM_IoT_DF_Corpora()
    iot_Names = get_IoT_Names()
    index = 0
    for iot_name in iot_Names:
        corpus_filename = iot_name + '.pickle'
        corpus_path = os.path.join(
            config.DIIM_IOT_CORPORA_STORE_PATH, corpus_filename)

        df_corpus = diim_IoT_DF_Corpora[index]
        corpus = load_DIIM_IoT_Corpus(corpus_path, iot_name, df_corpus)
        # print(corpus)
        # print('****')

        # add corpus to list
        DIIM_IoT_Corpora.append(corpus)
        index = index + 1


def load_DIIM_IoT_Corpus(corpus_path, iot_name, df_corpus):
    # load if already persited
    if (os.path.exists(corpus_path)):
        corpus = pd.read_pickle(corpus_path)
        logger.info(' loading of ' + corpus_path + ' done!')
    else:
        # create and persist corpus
        corpus = w2v_utils.convert_dataframe_to_documents(df_corpus)
        corpus_filename = iot_name + '.pickle'
        corpus_path = os.path.join(
            config.DIIM_IOT_CORPORA_STORE_PATH, corpus_filename)
        f = open(corpus_path, 'wb')
        pickle.dump(corpus, f)
        f.close()

    return corpus


# def load_DIIM_IoT_LSI_Models():
#     # check if files exits
#     # if they exists load
#     # else create, save and load
#     load_DIIM_IoT_Corpora()

#     index = 0
#     for corpus in DIIM_IoT_Corpora:
#         iot_name = IoT_Names[index]
#         index = index + 1
#         load_IoT_LSI_model(iot_name, corpus)


# def load_IoT_LSI_model(iot_name, documents):
#     lsi_path = os.path.join(config.DIIM_LSI_IOT_STORE_PATH,  iot_name)
#     if (os.path.exists(lsi_path)):
#         lsi_model = LsiModel.load(lsi_path)  # load model
#         logger.info(' loading of ' + iot_name + ' model  done!')

#     else:
#         # remove words that appear only once
#         frequency = defaultdict(int)
#         # new_corpus = w2v_utils.drop_empty_rowsandcolums(corpus)
#         #new_corpus = corpus.replace(to_replace='None', value=np.nan).dropna()
#         none_counter = 0
#         # remove common words and tokenize
#         stoplist = set('for a of the and to in'.split())
#         texts = [
#             [word for word in document.lower().split() if word not in stoplist]
#             for document in documents]

#         # print(texts[0])
#         # print(texts[1])
#         # print(texts[2])
#         # print(texts[3])
#         # print('No of nones', none_counter)
#         for text in texts:
#             for token in text:
#                 frequency[token] += 1

#         texts = [
#             [token for token in text if frequency[token] > 1]
#             for text in texts]

#         dictionary = corpora.Dictionary(texts)
#         # print(dictionary)

#         corpus = [dictionary.doc2bow(text) for text in texts]

#         tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
#         corpus_tfidf = tfidf[corpus]

#         # for doc in corpus_tfidf:
#         #     print(doc)

#         # initialize an LSI transformation
#         lsi_model = models.LsiModel(
#             corpus_tfidf, id2word=dictionary, num_topics=2)
#         # create a double wrapper over the original corpus: bow->tfi
#         corpus_lsi = lsi_model[corpus_tfidf]
#         lsi_model.save(lsi_path)
#     print(iot_name, lsi_model)
#     DIIM_IoT_LSI_Models.append(lsi_model)
#     # lsi_model.print_topics(2)
#     # for doc, as_text in zip(corpus_lsi, documents):
#     #     print(doc, as_text)


def initialize():
    load_DIIM_IoT_DF_Corpus_raw()
    load_DIIM_IoT_DF_Corpus()
    load_DIIM_IoT_Corpora()


# load_DIIM_IoT_Data()
initialize()
