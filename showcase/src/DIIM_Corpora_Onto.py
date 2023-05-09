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
import pandas as pd

# local imports
import DIIM_config as config
import DIIM_word2vec_utils as w2v_utils
import DIIM_word2vec_visualization as w2v_visual

# variables
logger = config.logger

# pre-processed tokens/words created while reading files in the ontology dir
DIIM_Ontos_DF_Corpora_raw = []  # list of dataframes

# pre-processed tokens in rows and columns
# removes empty rows
DIIM_Ontos_DF_Corpora = []  # list of dataframes

# names of the ontologies (directory names used to label an ontology)
Ontology_Names = []  # list of strings

DIIM_Ontos_Corpora = []  # list of strings extracted from the documents


def get_index_of_onto(onto_dir_name):
    if len(Ontology_Names) == 0:
        get_Ontology_Names()

    if onto_dir_name in Ontology_Names:
        return Ontology_Names.index(onto_dir_name)
    else:
        return None


def get_Ontology_Names():
    if len(Ontology_Names) == 0:
        for root, dirs, files in os.walk(config.ONTOS_DIR):
            for dir in dirs:
                # add the ontolgy name to the list
                Ontology_Names.append(dir)

    return Ontology_Names


def get_DIIM_Ontos_DF_Corpora_raw():
    if len(DIIM_Ontos_DF_Corpora_raw) == 0:
        load_DIIM_Ontos_DF_Corpora_raw()

    return DIIM_Ontos_DF_Corpora_raw


def get_DIIM_Ontos_DF_Corpora():
    if len(DIIM_Ontos_DF_Corpora) == 0:
        load_DIIM_Ontos_DF_Corpora()

    return DIIM_Ontos_DF_Corpora


def get_DIIM_Ontos_Corpora():
    if len(DIIM_Ontos_Corpora) == 0:
        load_DIIM_Ontos_Corpora()

    return DIIM_Ontos_Corpora

# TODO: refactor to w2vutil


def build_DIIM_Ontos_DF_Corpus_raw(root, onto_dir):
    DIIM_Ontology_Dir_path = Path(os.path.join(root, onto_dir))
    DIIM_DF_Corpus_raw_path = Path(config.DIIM_TMP_DIR_PATH, onto_dir)
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


def build_DIIM_Onto_DF_Corpus(root, onto_dir):
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
            #diim_df_corpus = load_DIIM_DataFrame_Corpus(root, dir)
            diim_df_corpus = build_DIIM_Onto_DF_Corpus(root, dir)
            print('Shape of cleaned Dataframe ',
                  dir, ': ', diim_df_corpus.shape)
            DIIM_Ontos_DF_Corpora.append(diim_df_corpus)


def load_DIIM_Ontos_DF_Corpora():
    # print('Length of ontology corpora list : ', len(DIIM_Ontos_DF_Corpora))

    if len(DIIM_Ontos_DF_Corpora) <= 0:
        build_DIIM_DF_Corpora(config.ONTOS_DIR)

    # print('Length of ontology corpora list after loading/building: ',
    #       len(DIIM_Ontos_DF_Corpora))


def load_DIIM_Ontos_DF_Corpora_raw():
    # print('Length of ontology corpora raw list: ',
    #       len(DIIM_Ontos_DF_Corpora_raw))

    if len(DIIM_Ontos_DF_Corpora_raw) <= 0:
        build_DIIM_Ontos_DF_Corpora_raw(config.ONTOS_DIR)

    # print('Length of ontology corpora raw list after loading/building: ',
    #       len(DIIM_Ontos_DF_Corpora_raw))


def load_DIIM_Ontos_Corpora():

    diim_Ontos_DF_Corpora = get_DIIM_Ontos_DF_Corpora()
    ontology_Names = get_Ontology_Names()
    index = 0
    for ontology_name in ontology_Names:
        corpus_filename = ontology_name + '.pickle'
        corpus_path = os.path.join(
            config.DIIM_ONTO_CORPORA_STORE_PATH, corpus_filename)

        df_corpus = diim_Ontos_DF_Corpora[index]
        corpus = load_DIIM_Onto_Corpus(corpus_path, ontology_name, df_corpus)
        # print(corpus)
        # print('****')

        # add corpus to list
        DIIM_Ontos_Corpora.append(corpus)
        index = index + 1


def load_DIIM_Onto_Corpus(corpus_path, ontology_name, df_corpus):
    # load if already persited
    if (os.path.exists(corpus_path)):
        corpus = pd.read_pickle(corpus_path)
        logger.info(' loading of ' + corpus_path + ' done!')
    else:
        # create and persist corpus
        corpus = w2v_utils.convert_dataframe_to_documents(df_corpus)
        corpus_filename = ontology_name + '.pickle'
        corpus_path = os.path.join(
            config.DIIM_ONTO_CORPORA_STORE_PATH, corpus_filename)
        f = open(corpus_path, 'wb')
        pickle.dump(corpus, f)
        f.close()

    return corpus


def initialize():
    load_DIIM_Ontos_DF_Corpora_raw()
    load_DIIM_Ontos_DF_Corpora()
    load_DIIM_Ontos_Corpora()


# initialize()
