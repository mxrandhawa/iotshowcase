###########################
# Docs: https://www.kaggle.com/code/jeffd23/visualizing-word-vectors-with-t-sne/notebook
#
# This script has following main routines
# 1) Ontology Dir ---convert & ---> Corpora raw: Words in ontologies as Dataframe (store them as pkl)
# 2) Dataframe Corpora raw ---preprocess---> Cleaned Dataframe Corpora
# 3) Cleaned Dataframe Corpora ---convert---> Word2Vec models of ontologies
# 4) Display scatter plots of Word2Vec of ontologies
##########################

# python imports
import logging
import os
from pathlib import Path
import string
import re
import sys

# matplot imports
import matplotlib.pyplot as plt
from matplotlib import pyplot
# numpy imports
import numpy as np
# pandas imports
import pandas as pd
# sklearn imports
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# gensim imports
from gensim.models import word2vec
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import preprocess_documents
from gensim import utils
from gensim.parsing.porter import PorterStemmer
from gensim.parsing import stem_text
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
import DIIM_word2vec_ontos as ontos_w2v
import DIIM_word2vec_iot_data as iot_w2v
import DIIM_word2vec_utils as w2v_utils
import S3O_Exporter as diim_sim_export


logger = config.logger

# variables
# logger = logging.getLogger(config.LOGAPPLICATION_NAME)
# EN_STOP_WORDS = stopwords.words('english')
# EN_STOP_WORDS.extend(['is', 'may', 'also', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
#                      'eight', 'nine', 'ten', 'across', 'among', 'beside', 'however', 'yet', 'within']+list(ascii_lowercase))

# # pre-processed tokens/words created while reading files in the ontology dir
# DIIM_Ontos_DF_Corpora_raw = []  # list of dataframes

# # pre-processed tokens in rows and columns
# # removes empty rows
# DIIM_Ontos_DF_Corpora = []  # list of dataframes

# # names of the ontologies (directory names used to label an ontology)
# Ontology_Names = []  # list of strings

# # word2vec models of ontologies
# DIIM_Ontos_W2V_Models = []  # list of word2vec models

# # pre-processed tokens/words created while reading files in the IoT dir
# DIIM_IoT_DF_Corpora_raw = []  # list of dataframes

# # pre-processed tokens in rows and columns
# # removes empty rows
# DIIM_IoT_DF_Corpora = []  # list of dataframes

# # names of the IoT (directory names used to label an IoT data)
# IoT_Names = []  # list of strings


# def tokenize_file(input_path):
#     lines = []
#     with open(input_path) as f:
#         lines = f.readlines()

#     # print('*** printing file content')
#     # for sentence in lines:
#     #     print(sent_tokenize(sentence))

#     # Preprocessing data to lowercase all words and remove single punctuation words
#     # data = preprocess_documents(lines)
#     # print(data)

#     data = []
#     for sent in lines:
#         new_sent = []
#         # clean sentence
#         sent = clean_sentence(sent)
#         # tokenize the sentence
#         words = word_tokenize(sent)
#         data.append(words)
#     #     for word in words:
#     #         new_word = word.lower()
#     #         if new_word[0] not in string.punctuation:
#     #             new_sent.append(new_word)
#     #     if len(new_sent) > 0:
#     #         data.append(new_sent)
#     # print(data)

#     return data


# def similar_words(ontology_name, model, words):
#     # Finding most similar words

#     for word in words:
#         print(ontology_name, " has following 10 words similar to " + word)
#         if word in model.wv:
#             similar_words = model.wv.most_similar(word, topn=10)
#             print(similar_words)
#         else:
#             print(ontology_name, ' Word2Vec model dosn\'t contain ', word)


# def drop_emptyrows(corpora_df):
#     # print(corpora_df.head)
#     index_of_emptyrows = []
#     # print(corpora_df.isna().sum())
#     # for i in range(len(corpora_df.index)):
#     nr_of_rows = corpora_df.shape[0]
#     nr_of_cols = corpora_df.shape[1]
#     print(range(nr_of_rows))
#     for i in range(nr_of_rows):
#         # count all nulls in the row
#         nr_of_Nulls = corpora_df.iloc[i].isnull().sum()
#         # if the nr of nulls equals nr of cols --> mark rows
#         if (nr_of_Nulls >= nr_of_cols - 1):  # remove lines with single word or no words at all
#             # print("Nan in row ", i, " : ", nr_of_Nulls)
#             index_of_emptyrows.append(i)

#     corpora_df = corpora_df.drop(index_of_emptyrows)

#     return corpora_df


# def replace_digits_with_space(string):
#     pattern = r'[0-9]'
#     # Match all digits in the string and replace them with an empty string
#     new_string = re.sub(pattern, ' ', string)
#     return new_string


# def replace_punctuation_with_space(string):
#     pattern = r'([^\s\w]|_)+'
#     # Match all digits in the string and replace them with an empty string
#     new_string = re.sub(pattern, ' ', string)
#     return new_string


def print_vocabulary(wv_model):
    wv_model_size = len(wv_model.wv.index_to_key)
    for index, word in enumerate(wv_model.wv.index_to_key):
        # print first 100 words
        if index == 100:
            break
        print(f"word #{index}/{wv_model_size} is {word}")


# def build_corpora(data, input_path):

#     more_sentences = tokenize_file(input_path)
#     for sent in more_sentences:
#         data.append(sent)
#     return data


def remove_punctuation(data):
    # list(list[str]) --> list(list[str])
    filtered_data = []
    for list_string in data:
        filtered_list = []
        for a_string in list_string:
            # treat url
            a_string.replace('.', '. ')
            a_string.replace('//', '// ')
            # remove puncuation
            new_string = a_string.translate(
                str.maketrans('', '', string.punctuation))
            filtered_list.append(new_string)
        filtered_data.append(filtered_list)
    print(filtered_data)
    return filtered_data


############# REFACTOR following code to utility file ############


# def setupLogger():
#     # create logger with 'dimm_application'
#     logger = logging.getLogger(config.LOGAPPLICATION_NAME)
#     logger.setLevel(logging.DEBUG)
#     # create file handler which logs even debug messages
#     fh = logging.FileHandler(config.LOGFILE_NAME)
#     fh.setLevel(logging.DEBUG)
#     # create console handler with a higher log level
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.ERROR)
#     # create formatter and add it to the handlers
#     # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     # formatter = logging.Formatter('%(levelname)s - %(message)s')
#     fh.setFormatter(formatter)
#     ch.setFormatter(formatter)
#     # add the handlers to the logger
#     logger.addHandler(fh)
#     logger.addHandler(ch)

    # logger.info('creating an instance of AnalyseJSON')
    # a = ajd.Auxiliary()
    # logger.info('created an instance of AnalyseJSON')


def load_DIIM_Data():
    logger.info('Trying to load stored DIIM Data from ' +
                       config.DIIM_STORE_PATH)
    # loads the ontology data
    ontos_w2v.load_DIIM_Ontos_Data()

    # loads IoT data
    iot_w2v.load_DIIM_IoT_Data()

    # w2v_utils.check_words_similarities_in_ontologies()

    logger.info('loading of DIIM DATA done!')


def visualize_DIIM_Data():
    logger.info('Starting visualization of DIIM data ...')
    ontos_w2v.visualize_DIIM_Ontos_W2V()
    logger.info('Visualization of DIIM data done!')

def load_IoT_Similarities():
    logger.info('Starting loading of DIIM data ...')
    iot_similarities = ontos_w2v.get_similarities_of_iot(
        iot_w2v.IoT_Names, iot_w2v.DIIM_IoT_DF_Corpora)
    logger.info("\nIoT Similarities to Ontologies: ")
    logger.info(iot_similarities)
    logger.info('Loading of DIIM data done!')



def get_IoT_Similarities():
    if (len(ontos_w2v.DIIM_Similarities_of_IOT_with_Ontos)) == 0:
        load_IoT_Similarities()

    return ontos_w2v.DIIM_Similarities_of_IOT_with_Ontos


def visualize_IoT_Similarities():
    ontos_w2v.visualize_IoT_Similarities()


def diim_execution_routine():
    logger.info("Current working directory: {0}".format(os.getcwd()))

    load_DIIM_Data()
    visualize_DIIM_Data()
    load_IoT_Similarities()

    # visualize as bar charts
    # working, but do not needed it because I have now RDF graph of all similairties
    # this function takes very long time and you need to comment
    # the ontology names in the list
    visualize_IoT_Similarities()

    # export only for a demo with max 2 similairty nodes
    diim_sim_export.save_DSO_Similarities_of_IOT_with_Ontos(
        get_IoT_Similarities(), diim_sim_export.DIIM_RDF_ALL, 1)

    # export all
    diim_sim_export.save_DSO_Similarities_of_IOT_with_Ontos(
        get_IoT_Similarities(), diim_sim_export.DIIM_RDF_ALL, sys.maxsize)


# main()
