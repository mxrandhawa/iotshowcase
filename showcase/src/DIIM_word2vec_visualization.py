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
import time

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

# variables
#logger = logging.getLogger(config.LOGAPPLICATION_NAME) # old code
logger = config.logger

EN_STOP_WORDS = stopwords.words('english')
EN_STOP_WORDS.extend(['is', 'may', 'also', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
                     'eight', 'nine', 'ten', 'across', 'among', 'beside', 'however', 'yet', 'within']+list(ascii_lowercase))

# pre-processed tokens/words created while reading files in the ontology dir
DIIM_Ontos_DF_Corpora_raw = []  # list of dataframes

# pre-processed tokens in rows and columns
# removes empty rows
DIIM_Ontos_DF_Corpora = []  # list of dataframes

# names of the ontologies (directory names used to label an ontology)
Ontology_Names = []  # list of strings

# word2vec models of ontologies
DIIM_Ontos_W2V_Models = []  # list of word2vec models

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


def tokenize_file(input_path):
    lines = []
    with open(input_path) as f:
        lines = f.readlines()

    # print('*** printing file content')
    # for sentence in lines:
    #     print(sent_tokenize(sentence))

    # Preprocessing data to lowercase all words and remove single punctuation words
    # data = preprocess_documents(lines)
    # print(data)

    data = []
    for sent in lines:
        new_sent = []
        # clean sentence
        sent = clean_sentence(sent)
        # tokenize the sentence
        words = word_tokenize(sent)
        data.append(words)
    #     for word in words:
    #         new_word = word.lower()
    #         if new_word[0] not in string.punctuation:
    #             new_sent.append(new_word)
    #     if len(new_sent) > 0:
    #         data.append(new_sent)
    # print(data)

    return data


def similar_words(ontology_name, model, words):
    # Finding most similar words

    for word in words:
        logger.info(ontology_name, " has following 10 words similar to " + word)
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=10)
            logger.info(similar_words)
        else:
            logger.info(ontology_name, ' Word2Vec model dosn\'t contain ', word)


def drop_emptyrows(corpora_df):
    # print(corpora_df.head)
    index_of_emptyrows = []
    # print(corpora_df.isna().sum())
    # for i in range(len(corpora_df.index)):
    nr_of_rows = corpora_df.shape[0]
    nr_of_cols = corpora_df.shape[1]
    logger.info(range(nr_of_rows))
    for i in range(nr_of_rows):
        # count all nulls in the row
        nr_of_Nulls = corpora_df.iloc[i].isnull().sum()
        # if the nr of nulls equals nr of cols --> mark rows
        if (nr_of_Nulls >= nr_of_cols - 1):  # remove lines with single word or no words at all
            # print("Nan in row ", i, " : ", nr_of_Nulls)
            index_of_emptyrows.append(i)

    corpora_df = corpora_df.drop(index_of_emptyrows)

    return corpora_df


def replace_digits_with_space(string):
    pattern = r'[0-9]'
    # Match all digits in the string and replace them with an empty string
    new_string = re.sub(pattern, ' ', string)
    return new_string


# def replace_punctuation_with_space(string):
#     pattern = r'([^\s\w]|_)+'
#     # Match all digits in the string and replace them with an empty string
#     new_string = re.sub(pattern, ' ', string)
#     return new_string


def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub(' ', val).lower()
    sentence = replace_digits_with_space(sentence)
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in EN_STOP_WORDS:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


def filter_stopwords(words):
    # remove it later
    filtered_sentence = []
    en_stopwords = set(stopwords.words('english'))
    for w in words:
        if w not in en_stopwords:
            filtered_sentence.append(w)

    Stem_words = []
    ps = PorterStemmer()
    for w in filtered_sentence:
        rootWord = ps.stem(w)
        Stem_words.append(rootWord)
    # print(filtered_sentence)
    # print(Stem_words)
    return filtered_sentence


def show_scatter_plot(model, words):
    X = model.wv[words]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    pyplot.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()


def print_vocabulary(wv_model):
    wv_model_size = len(wv_model.wv.index_to_key)
    for index, word in enumerate(wv_model.wv.index_to_key):
        # print first 100 words
        if index == 100:
            break
        logger.info(f"word #{index}/{wv_model_size} is {word}")


def build_corpora(data, input_path):

    more_sentences = tokenize_file(input_path)
    for sent in more_sentences:
        data.append(sent)
    return data


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
    logger.info(filtered_data)
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

#     # logger.info('creating an instance of AnalyseJSON')
#     # a = ajd.Auxiliary()
#     # logger.info('created an instance of AnalyseJSON')


def build_corpora_from_DIR():  # TODO delete it
    data = []
    # logger.info('starting with NLP of files ending with ' + fileType)
    # look up JSONFiles in current dir and analyze them
    logger.info('Looking for input data in dir ' + config.inputDir)
    for currentpath, folders, files in os.walk(config.inputDir + "\\ontos\SensorML"):
        for file in files:
            filePath = os.path.join(currentpath, file)
            logger.info('Creating corpus from ' + filePath)
            build_corpora(data, filePath)

    # Calling DataFrame constructor on list
    corpora_df = pd.DataFrame(data)
    return corpora_df


def build_corpora_from_DIR(ONTO_DIR_Path):
    data = []
    # logger.info('starting with NLP of files ending with ' + fileType)
    # look up JSONFiles in current dir and analyze them
    logger.info('Looking for input data in dir ', ONTO_DIR_Path)
    for currentpath, folders, files in os.walk(ONTO_DIR_Path):
     # for file in filter(lambda file: file.endswith('.txt'), files):
     # iot or examples for ontologies xml, rdf, csv, json
        for file in filter(lambda file: file.endswith('.owl')
                           or file.endswith('.rdf')
                           or file.endswith('.xml')
                           or file.endswith('.txt')
                           or file.endswith('.json')
                           or file.endswith('.csv')
                           or file.endswith('.html'), files):
            filePath = os.path.join(currentpath, file)
            logger.info('Creating corpus from ' + filePath)
            # data.append(build_corpora(filePath))
            build_corpora(data, filePath)

    # Calling DataFrame constructor on list
    corpora_df = pd.DataFrame(data)

    return corpora_df


def check_similarities(model, words):
    for word in words:
        similar_words(model, word)

# def save_DIIM_Dataframe_Corpus(diim_dataframe):


def build_DIIM_Ontos_DF_Corpus_raw(root, onto_dir):
    DIIM_Ontology_Dir_path = Path(os.path.join(root, onto_dir))
    DIIM_DF_Corpus_raw_path = Path(config.DIIM_DF_STORE_PATH, onto_dir)
    DIIM_DF_Corpus_raw_path = str(
        DIIM_DF_Corpus_raw_path) + "_DF_Corpus_raw.pkl"
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
        df_corpus_raw = build_corpora_from_DIR(DIIM_Ontology_Dir_path)
        logger.info(' saving newly created DF Corpus raw to ',
                    DIIM_DF_Corpus_raw_path)
        df_corpus_raw.to_pickle(DIIM_DF_Corpus_raw_path)

    # print(DIIM_Ontology_Dir_path)
    # print(df_corpus_raw.head())
    # print(df_corpus_raw.shape)
    return df_corpus_raw


def build_DIIM_Ontos_DF_Corpora_raw(ontos_dir):
    logger.info("Initializing DIIM Ontos corpus_raw from " + ontos_dir)
    # ontologies_dict_counter = 0  # counter for ontologies
    # number_of_read_files = 0
    for root, dirs, files in os.walk(ontos_dir):
        for dir in dirs:
            # add the ontolgy name to the list
            Ontology_Names.append(dir)
            diim_df_corpus_raw = build_DIIM_Ontos_DF_Corpus_raw(root, dir)
            logger.info('Shape of raw Dataframe ', dir,
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

        df_corpus = drop_emptyrows(
            DIIM_Ontos_DF_Corpora_raw[get_index_of_ontology(onto_dir)])
        logger.info('Size of ', onto_dir, ': ', df_corpus.shape)
        DIIM_Ontos_DF_Corpora.append(df_corpus)
        logger.info(' saving cleaned DF Corpus to ', DIIM_DF_Corpus_path)
        df_corpus.to_pickle(DIIM_DF_Corpus_path)

    return df_corpus


def build_DIIM_DF_Corpora(ontos_dir):
    for root, dirs, files in os.walk(ontos_dir):
        for dir in dirs:
            diim_df_corpus = load_DIIM_DataFrame_Corpus(root, dir)
            logger.info('Shape of cleaned Dataframe ',
                  dir, ': ', diim_df_corpus.shape)
            DIIM_Ontos_DF_Corpora.append(diim_df_corpus)


def load_DIIM_Ontos_DF_Corpora():
    logger.info('Length of ontology corpora list : ', len(DIIM_Ontos_DF_Corpora))

    if len(DIIM_Ontos_DF_Corpora) <= 0:
        build_DIIM_DF_Corpora(config.ONTOS_DIR)

    logger.info('Length of ontology corpora list after loading/building: ',
          len(DIIM_Ontos_DF_Corpora))


def load_DIIM_Ontos_DF_Corpora_raw():
    logger.info('Length of ontology corpora raw list: ',
          len(DIIM_Ontos_DF_Corpora_raw))

    if len(DIIM_Ontos_DF_Corpora_raw) <= 0:
        build_DIIM_Ontos_DF_Corpora_raw(config.ONTOS_DIR)

    logger.info('Length of ontology corpora raw list after loading/building: ',
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
            logger.info('Shape of cleaned Dataframe ',
                  dir, ': ', diim_df_corpus.shape)
            DIIM_Ontos_DF_Corpora.append(diim_df_corpus)


def load_DIIM_W2V_Models():
    logger.info('Length of DIIM Word2Vec models list: ',
          len(DIIM_Ontos_W2V_Models))

    if len(DIIM_Ontos_W2V_Models) <= 0:
        # load list of dataframes if it is empty
        if len(DIIM_Ontos_DF_Corpora) <= 0:
            load_DIIM_Ontos_DF_Corpora()

        for ontology_name in Ontology_Names:
            diim_w2c_model_path = Path(
                config.DIIM_WV_STORE_PATH, ontology_name)
            diim_w2c_model_path = str(diim_w2c_model_path) + "_w2v.model"

            if (os.path.exists(diim_w2c_model_path)):
                w2v_model = Word2Vec.load(diim_w2c_model_path)
            else:
                diim_df_corpus = DIIM_Ontos_DF_Corpora[Ontology_Names.index(
                    ontology_name)]
                corpora = diim_df_corpus.values.tolist()
                w2v_model = Word2Vec(sentences=corpora, vector_size=200,
                                     window=10, min_count=1, workers=4)
                # save the word2vec model of ontology
                w2v_model.save(diim_w2c_model_path)

            logger.info(ontology_name, ' Model Word2Vector size: ',
                  w2v_model.wv.vector_size)
            logger.info(ontology_name, ' Model Word2Vector Vocabulary size: ',
                  len(w2v_model.wv.index_to_key))
            DIIM_Ontos_W2V_Models.append(w2v_model)

    logger.info('Length of DIIM Word2Vec models list after loading/building: ',
          len(DIIM_Ontos_W2V_Models))


def check_words_similarities_in_ontologies():
    test_words = [
        'data',  'sensor', 'date', 'measurement', 'time', 'device']  # 'temp','water',
    for ontology_name in Ontology_Names:
        model = DIIM_Ontos_W2V_Models[Ontology_Names.index(ontology_name)]
        similar_words(ontology_name, model, test_words)
    # print('Model vector size: ', model.wv.vector_size)
    # print('Model Vocabulary size: ', len(model.wv.index_to_key))
    # print('Model Vocabulary size: ', type(model.wv.index_to_key))
    # model insights
    # print_vocabulary(model)
    # # print('*** printing model')
    # print(model.wv)

# visualization related methods


def show_scatter_plot_of_ontologies():
    test_words = [
        'data',  'sensor', 'date', 'measurement', 'time', 'device']  # 'temp','water',
    # visualization
    for model in DIIM_Ontos_W2V_Models:
        #show_scatter_plot(model, IoT_TEST_WORDS)
        words = []
        for word in IoT_TEST_WORDS:
            if word in model.wv:
                words.append(word)

        show_scatter_plot(model, words)
        #show_scatter_plot(model, test_words)

    # visualization
    show_scatter_plot(model, IoT_TEST_WORDS)
   # show_scatter_plot(model, model.wv.index_to_key[0:50])
    # show_scatter_plot(model, model.wv.index_to_key)

    tsne_plot(model)

# pd.options.mode.chained_assignment = None


def tsne_plot(ontology_name, model):
    # TODO working (not working yet see example??)
    # https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1
    "Creates and TSNE model and plots it"
    labels = []  # model.wv.index_to_key

    for index, word in enumerate(model.wv.index_to_key):
        # tokens.append(index)
        labels.append(word)
    total_labels = len(model.wv.index_to_key)
    labels_limit = total_labels

    # if labels_limit > 3000:
    #     # show 20% (50% taking too long)
    #     labels_limit = int((total_labels * 20)/100)

    logger.info(labels_limit)
    labels = labels[0:labels_limit]
    tokens = model.wv[labels]

    tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    model_title = ontology_name + ': showing words ' + \
        str(labels_limit) + '/' + str(len(model.wv.index_to_key))

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        # uncomment to show the words
        # plt.annotate(labels[i],
        #              xy=(x[i], y[i]),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom')
    plt.title(model_title)
    plt_img_filename_path = config.DIIM_IMG_STORE_PATH + ontology_name + \
        str(labels_limit) + "_from_" + str(total_labels) + '.png'
    plt.savefig(plt_img_filename_path)
    logger.info('Image saved to', plt_img_filename_path)
    # plt.savefig(plt_filename_path +'.png') # plot
    # plt.show()

# REFACTOR END


def show_barchart_of_similarities(ontology_iot_word_name, similarities):
    # close it
    plt.close('all')

    ontology_name, iot_name, label = w2v_utils.split_ontology_iot_word_name(ontology_iot_word_name)
    
    plot_title = 'Top 10 similar words for *' + label + \
        '* in Word2Vec: ' + ontology_name + '_' + iot_name

    words = []
    similarity_values = []
    for word, value in similarities:
        words.append(word)
        similarity_values.append(value)

    # initialize data as a dict/set object (this is 1 dimensional)
    data = {'Similarity': similarity_values}
    # singel colum dataframe to show bars
    df = pd.DataFrame(data, columns=['Similarity'], index=words)

    plt.style.use('ggplot')
    # Creating a figure with some fig size
    ax = df.plot.barh(figsize=(12, 6))

    # show values/annotations on the bars
    ax.bar_label(ax.containers[0])

    # place legend out of the bars-box
    horiz_offset = 1.
    vert_offset = 1.
    ax.legend(bbox_to_anchor=(horiz_offset, vert_offset))

    plt.title(plot_title)
    plt.ylabel('Words from *' + ontology_name + '* ontology')
    plt.xlabel('Similarity value')
    # plt.show()

    #  save the histograms to
    plt_img_filename_path = config.DIIM_IMG_STORE_PATH + \
        ontology_iot_word_name + '10_Sims.png'
    if os.path.exists(plt_img_filename_path):
         logger.info('Image exits at ', plt_img_filename_path)
    else:
        plt.savefig(plt_img_filename_path, bbox_inches='tight')
        logger.info('Image saved to ', plt_img_filename_path)

    # close it
    plt.close('all')

    


def show_histogram_of_similarities(ontology_iot_word_name, similarities):
    # TODO: fix Warning and Crashing
    # hecking similarities of  COSMO_BristolWaterQuality
    # Nr of combinations:  80
    # g:\MY\devl\DIIM\src\DIIM_Gensim_word2vec_visualization.py:620: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
    # plt.figure().clear()
    # G:\dev\bin\Python310\lib\site-packages\pandas\plotting\_matplotlib\core.py:345: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
    # fig = self.plt.figure(figsize=self.figsize)
    # Checking similarities of  DOLCE_BristolWaterQuality
    # Nr of combinations:  80
    # Fail to create pixmap with Tk_GetPixmap in TkImgPhotoInstanceSetSize
    words = []
    similarity_values = []

    for word, value in similarities:
        words.append(word)
        similarity_values.append(value)

    # better way to show bars with values
    df = pd.DataFrame(
        {'Similarity': similarity_values}, index=words)

    # similarity_values, index=words)
    # title_words = ontology_iot_word_name.split('_')
    # ontology_name = title_words[0]
    # iot_name = title_words[1]
    # label = title_words[2]
    # plot_title = label + ' from IoT ' + iot_name + \
    #     ' has following top 10 similar words in Ontology ' + ontology_name
    # df.plot(title=plot_title)

    #df.plot(title='Top 10 similar words: ' + ontology_iot_word_name)
    # fig = plt.figure()
    ax = df.plot.barh(figsize=(15, 15))
    # increase the size
    ax.bar_label(ax.containers[0])

    # Set x-axis label
    ax.set_xlabel("Similarity value", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    ax.set_ylabel('Top 10 similar words: ' + ontology_iot_word_name,
                  labelpad=20, weight='bold', size=12)

    #  save the histograms to
    plt_img_filename_path = config.outputDir + \
        ontology_iot_word_name + '10_Sims.png'
    plt.savefig(plt_img_filename_path, bbox_inches='tight')
    logger.info('Image saved to ', plt_img_filename_path)

    # uncomment below to show the plots
    # plt.show()

    # clears the plot
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def tsne_plot_of_similarties(ontology_iot_word_name, similarities):
    # TODO: not working, don't know how to setup tokens variable for tsne_model.fit_transform(tokens)
    "Creates and TSNE model and plots it"

    tokens = []  # similarity value

    # for index, word in enumerate(similarities):
    for word, value in similarities:
        labels = []  # word
        labels.append(value)
        # labels.append(word)

        tokens.append(labels)

    tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    model_title = ontology_iot_word_name + ': showing words top 10 similarities '

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        # uncomment to show the words
        # plt.annotate(labels[i],
        #              xy=(x[i], y[i]),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom')
    plt.title(model_title)
    plt_img_filename_path = config.outputDir + ontology_iot_word_name + '.png'
    plt.savefig(plt_img_filename_path)
    # plt.savefig(plt_filename_path +'.png') # plot
    plt.show()
    plt.clf()
