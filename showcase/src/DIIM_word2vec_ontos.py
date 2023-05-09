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
import pickle
import os
from pathlib import Path
import string
import re
import time
from unittest import skip

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
import DIIM_word2vec_utils as w2v_utils
import DIIM_word2vec_visualization as w2v_visual

# variables
logger = config.logger
# EN_STOP_WORDS = stopwords.words('english')
# EN_STOP_WORDS.extend(['is', 'may', 'also', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
#                      'eight', 'nine', 'ten', 'across', 'among', 'beside', 'however', 'yet', 'within']+list(ascii_lowercase))

# pre-processed tokens/words created while reading files in the ontology dir
DIIM_Ontos_DF_Corpora_raw = []  # list of dataframes

# pre-processed tokens in rows and columns
# removes empty rows
DIIM_Ontos_DF_Corpora = []  # list of dataframes

# names of the ontologies (directory names used to label an ontology)
Ontology_Names = []  # list of strings

# word2vec models of ontologies
DIIM_Ontos_W2V_Models = []  # list of word2vec models

# word2vec models of ontologies trained with iot
# set of word2vec ontology models that are trained with iot
# key = ontology_name + '_' + iot_name
# values = ord2vec models of ontology trained with iot
DIIM_Ontos_W2V_Models_IoT_Trained = {}

DIIM_Similarities_of_IOT_with_Ontos = {}

# # pre-processed tokens/words created while reading files in the IoT dir
# DIIM_IoT_DF_Corpora_raw = []  # list of dataframes

# # pre-processed tokens in rows and columns
# # removes empty rows
# DIIM_IoT_DF_Corpora = []  # list of dataframes

# # names of the IoT (directory names used to label an IoT data)
# IoT_Names = []  # list of strings

# # Test word from the IoT data on water quality monitoring
# IoT_TEST_WORDS = ['sensor', 'system', 'time', 'measurement', 'digital', 'platform',
#                   'actuator', 'data', 'value', 'property', 'determinand',
#                   'process', 'location', 'position', 'temperature', 'description', 'pH', 'Clorine']  # , 'temp'


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


# def clean_sentence(val):
#     "remove chars that are not letters or numbers, downcase, then remove stop words"
#     regex = re.compile('([^\s\w]|_)+')
#     sentence = regex.sub(' ', val).lower()
#     sentence = replace_digits_with_space(sentence)
#     sentence = sentence.split(" ")

#     for word in list(sentence):
#         if word in EN_STOP_WORDS:
#             sentence.remove(word)

#     sentence = " ".join(sentence)
#     return sentence


# def filter_stopwords(words):
#     # remove it later
#     filtered_sentence = []
#     en_stopwords = set(stopwords.words('english'))
#     for w in words:
#         if w not in en_stopwords:
#             filtered_sentence.append(w)

#     Stem_words = []
#     ps = PorterStemmer()
#     for w in filtered_sentence:
#         rootWord = ps.stem(w)
#         Stem_words.append(rootWord)
#     # print(filtered_sentence)
#     # print(Stem_words)
#     return filtered_sentence


# def show_scatter_plot(model, words):
#     X = model.wv[words]
#     pca = PCA(n_components=2)
#     result = pca.fit_transform(X)

#     pyplot.scatter(result[:, 0], result[:, 1])
#     for i, word in enumerate(words):
#         pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
#     pyplot.show()


# def print_vocabulary(wv_model):
#     wv_model_size = len(wv_model.wv.index_to_key)
#     for index, word in enumerate(wv_model.wv.index_to_key):
#         # print first 100 words
#         if index == 100:
#             break
#         print(f"word #{index}/{wv_model_size} is {word}")


# def build_corpora(data, input_path):

#     more_sentences = tokenize_file(input_path)
#     for sent in more_sentences:
#         data.append(sent)
#     return data


# def remove_punctuation(data):
#     # list(list[str]) --> list(list[str])
#     filtered_data = []
#     for list_string in data:
#         filtered_list = []
#         for a_string in list_string:
#             # treat url
#             a_string.replace('.', '. ')
#             a_string.replace('//', '// ')
#             # remove puncuation
#             new_string = a_string.translate(
#                 str.maketrans('', '', string.punctuation))
#             filtered_list.append(new_string)
#         filtered_data.append(filtered_list)
#     print(filtered_data)
#     return filtered_data


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

#     # logger.info('creating an instance of AnalyseJSON')
#     # a = ajd.Auxiliary()
#     # logger.info('created an instance of AnalyseJSON')


# def build_corpora_from_DIR():  # TODO delete it
#     data = []
#     # logger.info('starting with NLP of files ending with ' + fileType)
#     # look up JSONFiles in current dir and analyze them
#     logger.info('Looking for input data in dir ' + config.inputDir)
#     for currentpath, folders, files in os.walk(config.inputDir + "\\ontos\SensorML"):
#         for file in files:
#             filePath = os.path.join(currentpath, file)
#             logger.info('Creating corpus from ' + filePath)
#             build_corpora(data, filePath)

#     # Calling DataFrame constructor on list
#     corpora_df = pd.DataFrame(data)
#     return corpora_df


# def build_corpora_from_DIR(ONTO_DIR_Path):
#     data = []
#     # logger.info('starting with NLP of files ending with ' + fileType)
#     # look up JSONFiles in current dir and analyze them
#     logger.info('Looking for input data in dir ', ONTO_DIR_Path)
#     for currentpath, folders, files in os.walk(ONTO_DIR_Path):
#      # for file in filter(lambda file: file.endswith('.txt'), files):
#      # iot or examples for ontologies xml, rdf, csv, json
#         for file in filter(lambda file: file.endswith('.owl')
#                            or file.endswith('.rdf')
#                            or file.endswith('.xml')
#                            or file.endswith('.txt')
#                            or file.endswith('.json')
#                            or file.endswith('.csv')
#                            or file.endswith('.html'), files):
#             filePath = os.path.join(currentpath, file)
#             logger.info('Creating corpus from ' + filePath)
#             # data.append(build_corpora(filePath))
#             build_corpora(data, filePath)

#     # Calling DataFrame constructor on list
#     corpora_df = pd.DataFrame(data)

#     return corpora_df


# def check_similarities(model, words):
#     for word in words:
#         similar_words(model, word)

# # def save_DIIM_Dataframe_Corpus(diim_dataframe):


# TODO: refactor to w2vutil
def build_DIIM_Ontos_DF_Corpus_raw(root, onto_dir):
    DIIM_Ontology_Dir_path = Path(os.path.join(root, onto_dir))
    DIIM_DF_Corpus_raw_path = Path(config.DIIM_DF_STORE_PATH, onto_dir)
    DIIM_DF_Corpus_raw_path = str(
        DIIM_DF_Corpus_raw_path) + "_DF_Corpus_raw.pkl"
    DIIM_DF_Corpus_raw_path_CSV = str(
        DIIM_DF_Corpus_raw_path) + "_DF_Corpus_raw.csv"
    logger.info('Trying to load stored DIIM Dataframe corpus raw from ' +
                DIIM_DF_Corpus_raw_path)
    # df_corpus_raw = []
    # if a raw corpus of the ontology alread read and created load it
    if (os.path.exists(DIIM_DF_Corpus_raw_path)):
        df_corpus_raw = pd.read_pickle(DIIM_DF_Corpus_raw_path)
        logger.info(' loading of DIIM DATA done!')
    else:
        # else create a raw corpus of the ontology and save it
        logger.info(' file not found!')
        logger.info('Building Dataframe corpus raw from ' +
                    DIIM_DF_Corpus_raw_path)
        df_corpus_raw = w2v_utils.build_corpora_from_DIR(
            DIIM_Ontology_Dir_path)
        logger.info(' saving newly created DF Corpus raw to ',
                    DIIM_DF_Corpus_raw_path)
        df_corpus_raw.to_pickle(DIIM_DF_Corpus_raw_path)
        df_corpus_raw.to_csv(DIIM_DF_Corpus_raw_path_CSV)

    # print(DIIM_Ontology_Dir_path)
    # print(df_corpus_raw.head())
    # print(df_corpus_raw.shape)
    return df_corpus_raw


def build_DIIM_Ontos_DF_Corpora_raw(ontos_dir):  # TODO: refactor to w2vutil
    logger.info("Initializing DIIM Ontos corpus_raw from " + ontos_dir)
    # ontologies_dict_counter = 0  # counter for ontologies
    # number_of_read_files = 0
    for root, dirs, files in os.walk(ontos_dir):
        for dir in dirs:
            # add the ontolgy name to the list
            Ontology_Names.append(dir)
            diim_df_corpus_raw = build_DIIM_Ontos_DF_Corpus_raw(root, dir)
            print('Shape of raw Dataframe ', dir,
                  ': ', diim_df_corpus_raw.shape)
            DIIM_Ontos_DF_Corpora_raw.append(diim_df_corpus_raw)


def get_index_of_ontology(ontology_dir_name):
    if ontology_dir_name in Ontology_Names:
        return Ontology_Names.index(ontology_dir_name)
    else:
        return None


def load_DIIM_DataFrame_Corpus(root, onto_dir):
    DIIM_Ontology_Dir_path = Path(os.path.join(root, onto_dir))
    DIIM_DF_Corpus_path = Path(config.DIIM_DF_STORE_PATH, onto_dir)
    DIIM_DF_Corpus_path = str(DIIM_DF_Corpus_path) + "_DF_Corpus.pkl"
    DIIM_DF_Corpus_path_CSV = str(DIIM_DF_Corpus_path) + "_DF_Corpus.csv"

    logger.info('Trying to load stored DIIM Dataframe corpus from ' +
                DIIM_DF_Corpus_path)
    df_corpus = []

    # if a raw corpus of the ontology alread read and created load it
    if (os.path.exists(DIIM_DF_Corpus_path)):
        df_corpus = pd.read_pickle(DIIM_DF_Corpus_path)
        logger.info(' loading of DIIM DATA done!')
    else:
        # else create a raw corpus of the ontology and save it
        logger.info(' file not found!')
        logger.info('Building Dataframe corpus raw from ' +
                    DIIM_DF_Corpus_path)
        # Assure that list of DIIM_DataFrame_Corpora_raw has elements
        # and load_DIIM_DataFrame_Corpora_raw is executed
        if len(DIIM_Ontos_DF_Corpora_raw) <= 0:
            load_DIIM_Ontos_DF_Corpora_raw()
        df_corpus = DIIM_Ontos_DF_Corpora_raw[get_index_of_ontology(onto_dir)]

        df_corpus = df_corpus.drop_duplicates()
        df_corpus = w2v_utils.drop_empty_rowsandcolums(df_corpus)

        print('Size of ', onto_dir, ': ', df_corpus.shape)
        DIIM_Ontos_DF_Corpora.append(df_corpus)
        logger.info(' saving cleaned DF Corpus to ' + DIIM_DF_Corpus_path)
        df_corpus.to_pickle(DIIM_DF_Corpus_path)
        df_corpus.to_csv(DIIM_DF_Corpus_path_CSV)

    return df_corpus


def build_DIIM_DF_Corpora(ontos_dir):
    for root, dirs, files in os.walk(ontos_dir):
        for dir in dirs:
            diim_df_corpus = load_DIIM_DataFrame_Corpus(root, dir)
            print('Shape of cleaned Dataframe ',
                  dir, ': ', diim_df_corpus.shape)
            DIIM_Ontos_DF_Corpora.append(diim_df_corpus)


def load_DIIM_Ontos_DF_Corpora():
    print('Length of ontology corpora list : ', len(DIIM_Ontos_DF_Corpora))

    if len(DIIM_Ontos_DF_Corpora) <= 0:
        build_DIIM_DF_Corpora(config.ONTOS_DIR)

    print('Length of ontology corpora list after loading/building: ',
          len(DIIM_Ontos_DF_Corpora))


def load_DIIM_Ontos_DF_Corpora_raw():
    print('Length of ontology corpora raw list: ',
          len(DIIM_Ontos_DF_Corpora_raw))

    if len(DIIM_Ontos_DF_Corpora_raw) <= 0:
        build_DIIM_Ontos_DF_Corpora_raw(config.ONTOS_DIR)

    print('Length of ontology corpora raw list after loading/building: ',
          len(DIIM_Ontos_DF_Corpora_raw))

    # for diim_df_corpus_raw in DIIM_DataFrame_Corpora_raw:
    #     # print(type(diim_df_corpus_raw))
    #     print('Raw :', diim_df_corpus_raw.shape)
    #     corpora_df_corpus = drop_emptyrows(diim_df_corpus_raw)
    #     # print(corpora_df_corpus.head)
    #     # print(corpora_df.head)

    #     print('Cleaned: ', corpora_df_corpus.shape)
    #     DIIM_DataFrame_Corpora.append(corpora_df_corpus)


def build_DIIM_Word2Vec_Models(ontos_dir):
    for root, dirs, files in os.walk(ontos_dir):
        for dir in dirs:
            diim_df_corpus = load_DIIM_DataFrame_Corpus(root, dir)
            print('Shape of cleaned Dataframe ',
                  dir, ': ', diim_df_corpus.shape)
            DIIM_Ontos_DF_Corpora.append(diim_df_corpus)


def load_DIIM_W2V_Models():
    print('Length of DIIM Word2Vec models list: ',
          len(DIIM_Ontos_W2V_Models))

    if len(DIIM_Ontos_W2V_Models) <= 0:
        # load list of dataframes if it is empty
        if len(DIIM_Ontos_DF_Corpora) <= 0:
            load_DIIM_Ontos_DF_Corpora()

        for ontology_name in Ontology_Names:
            diim_w2v_model_path = Path(
                config.DIIM_WV_STORE_PATH, ontology_name)
            diim_w2v_model_path = str(diim_w2v_model_path) + "_w2v.model"

            if (os.path.exists(diim_w2v_model_path)):
                w2v_model = Word2Vec.load(diim_w2v_model_path)
            else:
                diim_df_corpus = DIIM_Ontos_DF_Corpora[Ontology_Names.index(
                    ontology_name)]
                corpora = diim_df_corpus.values.tolist()
                w2v_model = Word2Vec(sentences=corpora, vector_size=200,
                                     window=10, min_count=1, workers=4)
                # save the word2vec model of ontology
                w2v_model.save(diim_w2v_model_path)

            print(ontology_name, ' Model Word2Vector size: ',
                  w2v_model.wv.vector_size)
            print(ontology_name, ' Model Word2Vector Vocabulary size: ',
                  len(w2v_model.wv.index_to_key))
            DIIM_Ontos_W2V_Models.append(w2v_model)

    print('Length of DIIM Word2Vec models list after loading/building: ',
          len(DIIM_Ontos_W2V_Models))


def visualize_DIIM_Ontos_W2V():
    for ontology_name in Ontology_Names:
        model = DIIM_Ontos_W2V_Models[Ontology_Names.index(ontology_name)]
        w2v_visual.tsne_plot(ontology_name, model)


def build_DIIM_Similarities_of_IOT_with_Ontos(iot_names, iot_df_corpora):
    for i in range(len(iot_names)):
        iot_df_corpus = iot_df_corpora[i]
        words = set(iot_df_corpus.to_numpy().flatten())
        iot_name = iot_names[i]
        for j in range(len(Ontology_Names)):
            ontology_name = Ontology_Names[j]
            key = ontology_name + '_' + iot_name
            sim_values = []

            # get the trained model
            w2v_trained_model = DIIM_Ontos_W2V_Models_IoT_Trained[key]

            # build the path
            sim_file_name = key + ".pkl"
            diim__similarities_of_IOT_with_Ontos_path = Path(
                config.DIIM_SIM_STORE_PATH, sim_file_name)

            # check if model was saved and load it
            if (os.path.exists(diim__similarities_of_IOT_with_Ontos_path)):
                # Load data (deserialize)
                with open(diim__similarities_of_IOT_with_Ontos_path, 'rb') as handle:
                    sim_values = pickle.load(handle)
            else:
                # else create and save it for future iterations
                topn_limit = 10
                sim_values = w2v_utils.get_similar_words(
                    key, w2v_trained_model, words, topn_limit)
                # print(type(sim_values))
                # Store data (serialize)
                with open(diim__similarities_of_IOT_with_Ontos_path, 'wb') as handle:
                    pickle.dump(sim_values, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

            DIIM_Similarities_of_IOT_with_Ontos[key] = sim_values

            # print(type(sim_values))
            # print(sim_values)


def get_similarities_of_iot(iot_names, iot_df_corpora):
    #similarities_of_iot_with_ontologies = {}
    # if list is empty train w2v models with iot corpora
    if len(DIIM_Ontos_W2V_Models_IoT_Trained) <= 0:
        train_ontologies_with_iot(iot_names, iot_df_corpora)

    if len(DIIM_Similarities_of_IOT_with_Ontos) <= 0:
        build_DIIM_Similarities_of_IOT_with_Ontos(iot_names, iot_df_corpora)

    return DIIM_Similarities_of_IOT_with_Ontos


def train_ontologies_with_iot(iot_names, iot_df_corpora):

    for i in range(len(iot_names)):
        for j in range(len(Ontology_Names)):
            ontology_name = Ontology_Names[j]
            iot_name = iot_names[i]
            key = ontology_name + '_' + iot_name

            # build file path
            diim_w2v_model_trained_path = Path(
                config.DIIM_WV_STORE_PATH, key)
            diim_w2v_model_trained_path = str(
                diim_w2v_model_trained_path) + "_w2v.model"

            # check if model was saved and load it
            if (os.path.exists(diim_w2v_model_trained_path)):
                w2v_onto_model = Word2Vec.load(diim_w2v_model_trained_path)
            else:
                # else create and save it for future iterations
                w2v_onto_model = DIIM_Ontos_W2V_Models[j]
                iot_df_corpus = iot_df_corpora[i]
                sentences = iot_df_corpus.values.tolist()
            # w2v_trained_model.train(iot_df_corpus)
                w2v_onto_model.train(
                    sentences, total_examples=w2v_onto_model.corpus_count, epochs=30, report_delay=1)
                print(' Saving newly created W2V Ontology model trained with IoT to ',
                      diim_w2v_model_trained_path)
                w2v_onto_model.save(diim_w2v_model_trained_path)

            DIIM_Ontos_W2V_Models_IoT_Trained[key] = w2v_onto_model

        print('Total trained models : ', len(
            DIIM_Ontos_W2V_Models_IoT_Trained))


def visualize_IoT_Similarities():
    #sim_type = type(DIIM_Similarities_of_IOT_with_Ontos)
    #print('Sim object type ', sim_type)

    # for key, value in enumerate(DIIM_Similarities_of_IOT_with_Ontos):
    #     print(key, value)
    #w2v_visual.tsne_plot_of_similarties(key, value)
    nr_of_plots = 0
    for key in DIIM_Similarities_of_IOT_with_Ontos:
        #print(key, '->', DIIM_Similarities_of_IOT_with_Ontos[key])

        skip_ontos = ['COSMO_BristolWaterQuality',
                      'DOLCE_BristolWaterQuality',
                      'GO_BristolWaterQuality',
                      'GOIoTP_BristolWaterQuality',
                      'INSPIRE_BristolWaterQuality',
                      'OntoPlant_BristolWaterQuality',  # re create it
                      'OPO_BristolWaterQuality',
                      'SAREF_BristolWaterQuality',
                      'SensorML_BristolWaterQuality',
                      'SNN_BristolWaterQuality',
                      'SOSA_BristolWaterQuality',
                      'SWIM_BristolWaterQuality',
                      'WaterML_BristolWaterQuality',
                      'WatERPOntology_BristolWaterQuality',
                      'WHO_Drinking_BristolWaterQuality',
                      'WISDOM_BristolWaterQuality',
                      # second IoT
                      'COSMO_DrinkingWaterQualityMonitoring',
                      'DOLCE_DrinkingWaterQualityMonitoring',
                      'GO_DrinkingWaterQualityMonitoring',
                      'GOIoTP_DrinkingWaterQualityMonitoring',
                      'INSPIRE_DrinkingWaterQualityMonitoring',
                      'OntoPlant_DrinkingWaterQualityMonitoring',
                      'OPO_DrinkingWaterQualityMonitoring',
                      'SAREF_DrinkingWaterQualityMonitoring',
                      'SensorML_DrinkingWaterQualityMonitoring',
                      'SNN_DrinkingWaterQualityMonitoring',
                      'SOSA_DrinkingWaterQualityMonitoring',
                      'SWIM_DrinkingWaterQualityMonitoring',
                      'WaterML_DrinkingWaterQualityMonitoring',
                      'WatERPOntology_DrinkingWaterQualityMonitoring',
                      'WHO_Drinking_DrinkingWaterQualityMonitoring',
                      'WISDOM_DrinkingWaterQualityMonitoring',
                      # 3 IoT IndianRiverWaterQuality
                      'COSMO_IndianRiverWaterQuality',
                      'DOLCE_IndianRiverWaterQuality',
                      'GO_IndianRiverWaterQuality',
                      'GOIoTP_IndianRiverWaterQuality',
                      'INSPIRE_IndianRiverWaterQuality',
                      'OntoPlant_IndianRiverWaterQuality',
                      'OPO_IndianRiverWaterQuality',
                      'SAREF_IndianRiverWaterQuality',
                      'SensorML_IndianRiverWaterQuality',
                      'SNN_IndianRiverWaterQuality',
                      'SOSA_IndianRiverWaterQuality',
                      'SWIM_IndianRiverWaterQuality',
                      'WaterML_IndianRiverWaterQuality',
                      'WatERPOntology_IndianRiverWaterQuality',
                      'WHO_Drinking_IndianRiverWaterQuality',
                      'WISDOM_IndianRiverWaterQuality',
                      # 4 KaaIoTData
                      'COSMO_KaaIoTData',
                      'DOLCE_KaaIoTData',
                      'GO_KaaIoTData',
                      'GOIoTP_KaaIoTData',
                      'INSPIRE_KaaIoTData',
                      'OntoPlant_KaaIoTData',
                      'OPO_KaaIoTData',
                      'SAREF_KaaIoTData',
                      'SensorML_KaaIoTData',
                      'SNN_KaaIoTData',
                      'SOSA_KaaIoTData',
                      'SWIM_KaaIoTData',
                      'WaterML_KaaIoTData',
                      'WatERPOntology_KaaIoTData',
                      'WHO_Drinking_KaaIoTData',
                      'WISDOM_KaaIoTData',
                      # 5 NHFD
                      'COSMO_NHFD',
                      'DOLCE_NHFD',
                      'GO_NHFD',
                      'GOIoTP_NHFD',
                      'INSPIRE_NHFD',
                      'OntoPlant_NHFD',
                      'OPO_NHFD',
                      'SAREF_NHFD',
                      'SensorML_NHFD',
                      'SNN_NHFD',
                      'SOSA_NHFD',
                      'SWIM_NHFD',
                      'WaterML_NHFD',
                      'WatERPOntology_NHFD',
                      'WHO_Drinking_NHFD',
                      'WISDOM_NHFD'
                      ]
        skip_ontos = [] # try it without skipping
        if key in skip_ontos:
            print('skiping ', key)
        else:
            print('Checking similarities of ', key)
            print('Nr of Ontoloy to IoT combinations: ',
                  len(DIIM_Similarities_of_IOT_with_Ontos))
            similarities_dict = DIIM_Similarities_of_IOT_with_Ontos[key]
            print('Nr of words checked for similarities ', len(similarities_dict))
            for sim_key in similarities_dict:

                w2v_visual.show_barchart_of_similarities(
                    sim_key, similarities_dict[sim_key])

            # nr_of_plots =  nr_of_plots +1
            # print('Number of plots created ', str(nr_of_plots))

        # print('delay for clean closing of the plot figures')
        # i = 0
        # while i < 5000:
        #     i = i+1
        # print('Sleeping for 5 seconds!')
        # time.sleep(5)







def load_DIIM_Ontos_Data():
    load_DIIM_Ontos_DF_Corpora_raw()
    load_DIIM_Ontos_DF_Corpora()
    load_DIIM_W2V_Models()
