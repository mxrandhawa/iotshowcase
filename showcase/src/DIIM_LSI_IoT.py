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

from pprint import pprint
from gensim.models import LsiModel
from gensim import models
from gensim import corpora
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim import corpora
import os


# local imports
import DIIM_config as config
import DIIM_Corpora_IoT as corpora_iot
# variables
logger = config.logger

DIIM_IoT_LSI_Models = []
DIIM_IoT_Dictionaries = []
DIIM_IOT_Doc2BoW_Models = [] 


def get_DIIM_IoT_LSI_Models():
    if len(DIIM_IoT_LSI_Models) == 0:
        load_DIIM_IoT_LSI_Models()

    return DIIM_IoT_LSI_Models


def get_DIIM_IoT_Dictionaries():
    if len(DIIM_IoT_Dictionaries) == 0:
        load_DIIM_IoT_LSI_Models()

    return DIIM_IoT_Dictionaries

def get_DIIM_IOT_Doc2BoW_Models():
    if len(DIIM_IOT_Doc2BoW_Models) ==0:
        load_DIIM_IoT_LSI_Models()

    return DIIM_IOT_Doc2BoW_Models




def load_DIIM_IoT_LSI_Models():
    diim_IoT_Corpora = corpora_iot.get_DIIM_IoT_Corpora()
    iot_Names = corpora_iot.get_IoT_Names()

    index = 0
    for corpus in diim_IoT_Corpora:
        iot_name = iot_Names[index]
        index = index + 1
        load_IoT_LSI_and_BoW_model_and_dictionary(iot_name, corpus)


def load_IoT_LSI_and_BoW_model_and_dictionary(iot_name, documents):
    lsi_path = os.path.join(config.DIIM_LSI_IOT_STORE_PATH,  iot_name)
    dict_file = iot_name + '.txt'
    dict_path = os.path.join(config.DIIM_DICT_IOT_STORE_PATH,  dict_file)

    if (os.path.exists(lsi_path)):
        # load model from the file
        logger.info('Loading model from  ' + lsi_path)
        lsi_model = LsiModel.load(lsi_path)  # load model
        logger.info(' Loading of saved ' + iot_name + ' model done!')
        # load dictionaory from the file
        logger.info('Loading dictionary from  ' + dict_path)
        # load dict
        loaded_dct = Dictionary.load_from_text(dict_path)
        logger.info(' Loading of saved ' + iot_name + ' dict done!')
    else:
        logger.info('Createing new LSI model and dictionary for  ' + iot_name)
        lsi_model, loaded_dct = build_IoT_LSI_model_and_dictionary(
            iot_name, lsi_path, dict_path, documents)

    #print(iot_name, lsi_model)
    DIIM_IoT_LSI_Models.append(lsi_model)
    DIIM_IoT_Dictionaries.append(loaded_dct)


def build_IoT_LSI_model_and_dictionary(iot_name, lsi_path, dict_path, documents):
    # remove words that appear only once
    frequency = defaultdict(int)
    # remove common words and tokenize
    stoplist = set('for a of the and to in null'.split())
    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents]

    # for text in texts:
    #     for token in text:
    #         frequency[token] += 1

    # texts = [
    #     [token for token in text if frequency[token] > 1]
    #     for text in texts]

    # Dictionary
    dictionary = corpora.Dictionary(texts)
    dictionary.save_as_text(dict_path)
    logger.info('New dictionary  ' + iot_name + ' saved to ' + dict_path)

    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]

    # initialize an LSI transformation
    lsi_model = models.LsiModel(
        corpus_tfidf, id2word=dictionary, num_topics=2)
    # create a double wrapper over the original corpus: bow->tfi
    corpus_lsi = lsi_model[corpus_tfidf]
    lsi_model.save(lsi_path)
    logger.info('New LSI model for  ' + iot_name + ' saved to ' + lsi_path)

    return lsi_model, dictionary


def print_topics():
    index = 0
    print('Top 10 topics for ontology models ')
    ontology_Names = corpora_iot.get_IoT_Names()
    for lsi_model in DIIM_IoT_LSI_Models:
        print('\nTop 10 topics from ' + ontology_Names[index] + ':')
        print(type(lsi_model))
        # print(lsi_model.get_topics())
        pprint(lsi_model.print_topics())
        index = index + 1


def print_dictionaries():
    diim_IoT_Dictionaries = get_DIIM_IoT_Dictionaries()
    index = 0
    iot_names = corpora_iot.get_IoT_Names()
    for dictionary in diim_IoT_Dictionaries:
        print('Dictionary deatils of :', iot_names[index])
        print(dictionary)
        pprint(dictionary.token2id)
        # for token2id in token2ids:
        #     print(token2id)
        print()
        index = index + 1


def initialize():
    load_DIIM_IoT_LSI_Models()
    print_topics()
    # print_dictionaries()


initialize()


# def main():
#     vec_bow = dictionary.doc2bow(doc.lower().split())
#     vec_lsi = lsi[vec_bow]  # convert the query to LSI space
#     print(vec_lsi)


# main()
