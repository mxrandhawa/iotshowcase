# this is working 28/03/2022
from distutils.log import info
from pathlib import Path
import PyPDF2
import os
import string
import gensim
from gensim import similarities
from gensim import models
from gensim import corpora
from collections import defaultdict
from gensim.corpora import Dictionary
import pprint
import re
import json
import logging
from gensim import corpora
from gensim import similarities
from matplotlib.pyplot import title
from numpy import number
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt

# local imports
import DIIM_config as config
logger = logging.getLogger(config.LOGAPPLICATION_NAME)
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Core concepts in Gensim
# Document: some text.
# Corpus: a collection of documents.
# Vector: a mathematically convenient representation of a document.
# Model: an algorithm for transforming vectors from one representation to another.


PROJECT_DIR = os.getcwd() + "\\"
indexFilepath = PROJECT_DIR + 'res\\\data\\tmp\\diim_ontos.index'
GOIoTP_DIR = PROJECT_DIR + 'res\\data\\input\\ontos\\domain\\GOIoTP'
ONTOS_DIR = PROJECT_DIR + 'res\\data\\input\\ontos'

DIIM_STORE_PATH = PROJECT_DIR + 'res\\data\\store'

# path to save the DIIM Corpus raw
DIIM_CORPUS_RAW_PATH = DIIM_STORE_PATH + '\\DIIM_CORPUS_RAW.json'
DIIM_corpus_raw = []  # processed model from ontologies

# path to save the DIIM Corpus processed
DIIM_CORPUS_PROCESSED_PATH = DIIM_STORE_PATH + '\\DIIM_CORPUS_PROCESSED.json'
DIIM_corpus_processed = []  # processed model from ontologies

# path to save the DIIM dictionary
DIIM_DICT_PATH = DIIM_STORE_PATH + '\\DIIM_DICT.json'
DIIM_DICT = corpora.Dictionary("")  # empty dictionary

# GOIoTP_D1 = ''
ontologies_dict = {}  # set of names of ontologies
DIIM_ONTOS_DICT_PATH = DIIM_STORE_PATH + '\\DIIM_ONTOS_DICT.json'
ontologies_dict_counter = 0  # counter for ontologies

# DIIM_Text_Corpus = []  # list of documents to train the model, that will be ontologies
# DIIM_Corpus_for_training = []
# DIIM corpus represented as a Bag of Words vector
DIIM_BoW_corpus = DIIM_DICT.doc2bow([''])  # empty BoW representation
DIIM_BOW_CORPUS_PATH = DIIM_STORE_PATH + '\\DIIM_BOW_CORPUS.mm'
# DIIM_BOW = []  # DIIM represented as a Bag of Words vector
# initialize with empty corpus and dictionary
# DIIM_LSI = models.LsiModel(DIIM_CORPUS, id2word=DIIM_DICT)
stoplist = set(
    'a all and are at an as any \
     be by both but\
     can \
     has have had\
     only thus\
     shall if its \
     must may for  of the  to in or from have was their from does not is an we it other ie eg how why such out found more \
     this that then \
     which when what'.split(' '))

#####################
# IoT Data related variables
#############################
iot_data_dict = {}  # set of iot data documents, which will be used to find similarity
iot_data_dict_counter = 0  # counter for iot data documents
IOT_DATASET_PATH = PROJECT_DIR + 'res\\data\\input\\iot\\'
IoT_dataset = {}  # key is filename, value is iot data in the file


iot_data = "Meta data: Name:S1, Description: The sensor measures water temperature inFahrenheit, serial:00-14-22-01-23-45, model: BFG9000, mac:50:8c:b1:77:e8:e6, latitude:51.75543, longitude:-1.03248 Measurement data:4baa-a2ff-8741efad4e63: temp:timestamp:2021-08-09T17:01:28.796Z, values: value :20, timestamp:2021-08-09T17:01:38.792Z, values:value:24"
IoT_Data_BoW = []


def initialize_DIIM_corpus_raw():
    # ToFix: if a directory does not contains any filter relevent file, e.g.
    # txt or owl it leads to uneven size of ontos dictionary and DIIM corpus size
    logger.info("Initializing DIIM_corpus_raw from " + ONTOS_DIR)
    ontologies_dict_counter = 0  # counter for ontologies
    number_of_read_files = 0
    for root, dirs, files in os.walk(ONTOS_DIR):
        document = ""
        document_dir_name = ''
        # for file in filter(lambda file: file.endswith('.txt'), files):
        for file in filter(lambda file: file.endswith('.owl') or file.endswith('.xml') or file.endswith('.txt') or file.endswith('.html'), files):
            # for file in filter(lambda file: file.endswith('OGC_SensorML21.txt'), files):
            # read the entire document, as one big string
            document_path = Path(os.path.join(root, file))
            document += open(document_path, encoding="utf-8").read()
            number_of_read_files += 1
            if (len(document_dir_name) == 0):
                # get then name of directory
                document_dir_names = str(document_path.parent).split("\\")
                document_dir_name = document_dir_names[len(
                    document_dir_names)-1]
            logger.info("Adding " + file + " from Dir " +
                        document_dir_name + " to DIIM_corpus_raw")

        if (len(document_dir_name) > 0 and len(document) > 0):
            global ontologies_dict
            # add name of directory as ontology name to ontologies dictionary
            ontologies_dict[ontologies_dict_counter] = document_dir_name
            logger.info("Adding " + document_dir_name + " with key " +
                        str(ontologies_dict_counter) + " to ontologies dictionary")
            ontologies_dict_counter = 1 + ontologies_dict_counter
            # append aggregated text of all files of directory (ontology) to the corpus
            DIIM_corpus_raw.append(document)
            logger.info("DIIM_corpus_raw initialized with " +
                        str(len(DIIM_corpus_raw)) + " documents")
    logger.info("Ontologies read as direcotry names ..")
    for item in ontologies_dict.items():
        logger.info(item)


def preprocess_DIIM_document_string(document_string):
    # remove  puncuation characters from ontologies
    punctuation_chars = re.escape(string.punctuation)
    # print(string.punctuation)
    document_string = re.sub(r'['+punctuation_chars+']', ' ', document_string)
    punctuation_chars = '“'
    document_string = re.sub(r'['+punctuation_chars+']', '', document_string)
    punctuation_chars = '”'
    document_string = re.sub(r'['+punctuation_chars+']', '', document_string)
    # remove digits
    pattern = r'[0-9]'
    # Match all digits in the string and replace them with an empty string
    document_string = re.sub(pattern, '', document_string)

    # do not count words less then 2 charaters
    document_string = re.sub(r'\b\w{1,2}\b', '', document_string)
    return document_string


def preprocess_DIIM_corpus_raw():
    logger.info("Preprocessing DIIM_corpus_raw ... ")
    logger.info("Preprocessed DIIM_CORPUS has length " +
                str(len(DIIM_corpus_raw)))
    # Lowercase each document, split it by white space and filter out stopwords
    preprocessed_diim_documents = [preprocess_DIIM_document_string(document)
                                   for document in DIIM_corpus_raw]
    # Lowercase each document, split it by white space and filter out stopwords
    preprocessed_diim_documents = [[word for word in document.lower().split() if word not in stoplist]
                                   for document in preprocessed_diim_documents]

    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in preprocessed_diim_documents:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1]
                        for text in preprocessed_diim_documents]
    # print(processed_corpus)
    logger.info("Preprocessing of DIIM_corpus_raw done !")
    logger.info("Preprocessed DIIM_CORPUS has length " +
                str(len(processed_corpus)))
    return processed_corpus


def create_DIIM_corpus_processed():
    logger.info("creating DIIM corpus processed ...")
    global DIIM_corpus_processed
    DIIM_corpus_processed = preprocess_DIIM_corpus_raw()
    logger.info("DIIM corpus processed created!")
    return DIIM_corpus_processed


def create_DIIM_DICT():
    global DIIM_DICT
    DIIM_DICT = corpora.Dictionary(DIIM_corpus_processed)
    # print(DIIM_DICT)
    return DIIM_DICT


def keep_word_with_frequency(words_list, frequencyLimit):
    frequency = defaultdict(int)
    for token in words_list:
        # for token in word:
        frequency[token] += 1

    # Only keep words that appear more than once frequencyLimit] = 1
    filterd_list = [
        token for token in words_list if frequency[token] > frequencyLimit]
    # pprint.pprint(DIIM_corpus_processed)
    return filterd_list


def store_DIIM_BoW_corpus():
    # print("*** before storing DIIM_BoW_Corpus")
    # print(type(DIIM_BoW_corpus))
    # print(str(len(DIIM_BoW_corpus)))
    # encoded_DIIM_BoW_Vec = json.dumps(DIIM_BoW_corpus)
    # jsonFile = open(DIIM_BOW_CORPUS_PATH, "w")
    # jsonFile.write(encoded_DIIM_BoW_Vec)
    # jsonFile.close()
    # DIIM_BoW_corpus.save(DIIM_BOW_CORPUS_PATH)
    corpora.MmCorpus.serialize(DIIM_BOW_CORPUS_PATH, DIIM_BoW_corpus)
    logger.info('DIIM_BoW_corpus stored to  ' +
                DIIM_BOW_CORPUS_PATH)


def load_DIIM_BoW_corpus():
    global DIIM_BoW_corpus
    print('Size DIIM BoW Corpus', len(DIIM_BoW_corpus))
    print('DIIM BoW Corpus ', DIIM_BoW_corpus)

    file_exists = Path(DIIM_BOW_CORPUS_PATH).exists()
    # if file does not exits do not try to load
    if not file_exists:
        logger.info("Cann't load DIIM_BOW_CORPUS because file " +
                    DIIM_BOW_CORPUS_PATH + " doesn't exsit")
        return False

    logger.info(
        'Trying to load DIIM_corpus_processed from ' + DIIM_BOW_CORPUS_PATH)
    # fileObject = open(DIIM_BOW_CORPUS_PATH, "r")
    # jsonContent = fileObject.read()

    # DIIM_BoW_corpus = json.loads(jsonContent)
    DIIM_BoW_corpus = corpora.MmCorpus(DIIM_BOW_CORPUS_PATH)
    print("*** afer loading DIIM_BoW_Corpus")
    print(type(DIIM_BoW_corpus))
    print(str(len(DIIM_BoW_corpus)))
    logger.info(
        'DIIM_BoW_corpus loaded !')

    print('Size DIIM BoW Corpus', len(DIIM_BoW_corpus))
    print('DIIM BoW Corpus ', DIIM_BoW_corpus)
    for line in DIIM_BoW_corpus.index:
        print(line)

    return True


def store_DIIM_corpus_processed():
    #    print(len(DIIM_corpus_processed))
 #   print(len(ontologies_dict))
    encoded_diim_corpus_processed = json.dumps(DIIM_corpus_processed)
    jsonFile = open(DIIM_CORPUS_PROCESSED_PATH, "w")
    jsonFile.write(encoded_diim_corpus_processed)
    jsonFile.close()
    logger.info('DIIM_corpus_processed stored to  ' +
                DIIM_CORPUS_PROCESSED_PATH)


def load_DIIM_corpus_processed():
    file_exists = exists(DIIM_CORPUS_PROCESSED_PATH)
    # if file does not exits do not try to load
    if not file_exists:
        logger.info("Cann't load DIIM_corpus_processed because file " +
                    DIIM_CORPUS_PROCESSED_PATH + " doesn't exsit")
        return False

    logger.info(
        'Trying to load DIIM_corpus_processed from ' + DIIM_CORPUS_PROCESSED_PATH)
    fileObject = open(DIIM_CORPUS_PROCESSED_PATH, "r")
    jsonContent = fileObject.read()
    global DIIM_corpus_processed
    DIIM_corpus_processed = json.loads(jsonContent)
    logger.info(
        'DIIM_corpus_processed loaded !')
    return True


def store_DIIM_corpus_raw():
    global DIIM_corpus_raw
    encoded_diim_corpus_raw = json.dumps(DIIM_corpus_raw)
    jsonFile = open(DIIM_CORPUS_RAW_PATH, "w")
    jsonFile.write(encoded_diim_corpus_raw)
    jsonFile.close()
    logger.info('DIIM_corpus_raw stored to  ' +
                DIIM_CORPUS_RAW_PATH)


def load_DIIM_corpus_raw():
    file_exists = exists(DIIM_CORPUS_RAW_PATH)
    # if file does not exits do not try to load
    if not file_exists:
        logger.info("Cann't load DIIM_corpus_raw because file " +
                    DIIM_CORPUS_RAW_PATH + " doesn't exsit")
        return False

    logger.info('Trying to load DIIM_corpus_raw from ' + DIIM_CORPUS_RAW_PATH)
    fileObject = open(DIIM_CORPUS_RAW_PATH, "r")
    jsonContent = fileObject.read()
    global DIIM_corpus_raw
    DIIM_corpus_raw = json.loads(jsonContent)
    logger.info('Loading of DIIM_corpus_raw done!')
    return True


def store_DIIM_DICT():
    DIIM_DICT.save(DIIM_DICT_PATH)
    logger.info('DIIM DICT stored to' + DIIM_DICT_PATH)


def load_DIIM_DICT():
    file_exists = exists(DIIM_DICT_PATH)
    # if file does not exits do not try to load
    if not file_exists:
        logger.info("Cann't load DIIM_DICT because file " +
                    DIIM_DICT_PATH + " doesn't exsit")
        return False
    global DIIM_DICT
    logger.info('Trying to load stored DIIM DICT from ' + DIIM_DICT_PATH)
    DIIM_DICT = corpora.Dictionary.load(DIIM_DICT_PATH)
    logger.info('Loading of stored DIIM DICT done!')
    return True


def load_DIIM_ontos_dict():
    file_exists = exists(DIIM_ONTOS_DICT_PATH)
    # if file does not exits do not try to load
    if not file_exists:
        return False

    logger.info('Trying to load DIIM_ontos_dict from ' + DIIM_ONTOS_DICT_PATH)
    # fileObject = open(DIIM_ONTOS_DICT_PATH, "r")
    # jsonContent = fileObject.read()
    global ontologies_dict
    # ontologies_dict = json.loads(jsonContent)
    dict_file = open(DIIM_ONTOS_DICT_PATH, "r")
    ontologies_dict = json.load(dict_file, object_hook=jsonKeys2int)

    print(ontologies_dict)
    logger.info('Loading of DIIM_Ontos_dict done!')
    return True


def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k): v for k, v in x.items()}
    return x


def jsonKV2int(x):
    import unicodedata
    if isinstance(x, dict):
        return {int(k): (int(v) if isinstance(v, unicode) else v) for k, v in x.items()}
    return x


def store_DIIM_ontos_dict():
    dict_file = open(DIIM_ONTOS_DICT_PATH, "w")
    json.dump(ontologies_dict, dict_file)
    dict_file.close()

    # encoded_diim_ontos_dict = json.dumps(ontologies_dict)
    # jsonFile = open(DIIM_ONTOS_DICT_PATH, "w")
    # jsonFile.write(encoded_diim_ontos_dict)
    # jsonFile.close()
    logger.info('DIIM_ontos_dict stored to  ' +
                DIIM_ONTOS_DICT_PATH)


def store_DIIM_data():
    store_DIIM_corpus_raw()
    store_DIIM_corpus_processed()
    store_DIIM_DICT()
    store_DIIM_ontos_dict()
    store_DIIM_BoW_corpus()


def load_DIIM_data():
    logger.info('Trying to load stored DIIM DATA from ' + DIIM_STORE_PATH)
    # loads the data
    flag = True
    flag = flag and load_DIIM_corpus_raw()
    flag = flag and load_DIIM_corpus_processed()
    flag = flag and load_DIIM_DICT()
    flag = flag and load_DIIM_ontos_dict()
    flag = flag and load_DIIM_BoW_corpus()
    if flag:
        logger.info('loading of DIIM DATA done!')
    return flag


def create_vector_space(iot_data):
    iot_data_processed = preprocess_DIIM_document_string(iot_data)
    # Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.
    global IoT_Data_BoW
    IoT_Data_BoW = DIIM_DICT.doc2bow(iot_data_processed.lower().split())
    for token_id, token_count in IoT_Data_BoW:
        logger.info("Token \'" + DIIM_DICT[token_id] +
                    "\' occured " + str(token_count) + ' time(s) in IoT Data')
    return IoT_Data_BoW


def create_IoT_BoW_model(iot_data):
    iot_data_processed = preprocess_DIIM_document_string(iot_data)
    #print("IoT Data Processed ...")
    # print(iot_data_processed)
    # print('')
    # Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.
    global IoT_Data_BoW
    IoT_Data_BoW = DIIM_DICT.doc2bow(iot_data_processed.lower().split())
    for token_id, token_count in IoT_Data_BoW:
        logger.info("Token \'" + DIIM_DICT[token_id] +
                    "\' occured " + str(token_count) + ' time(s) in IoT Data')
    return IoT_Data_BoW


def create_DIIM_BoW_Vec():
    logger.info("creating BoW form DIIM_corpus processed ...")
    global DIIM_BoW_corpus
    # print(len(DIIM_corpus_processed))
    # print(DIIM_DICT)
    DIIM_BoW_corpus = [DIIM_DICT.doc2bow(text)
                       for text in DIIM_corpus_processed]
    logger.info("created BoW form DIIM_corpus has length " +
                str(len(DIIM_BoW_corpus)))
    return DIIM_BoW_corpus


def run_gensim_coreconcepts_ex():
    lsi = models.LsiModel(DIIM_BoW_corpus, id2word=DIIM_DICT,)
    vec_lsi = lsi[IoT_Data_BoW]  # convert the query to LSI space
    logger.info(vec_lsi)
    # transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(lsi[DIIM_BoW_corpus])
    indexFilepath
    index.save(indexFilepath)
    index = similarities.MatrixSimilarity.load(indexFilepath)
    # print(create_DIIM_BoW_Vec())
    # store_DIIM_data()
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    # print (document_number, document_similarity) 2-tuples
    # for document_number, document_similarity in enumerate(sims):
    #    logger.info("Ontology " + str(document_number) +
    #               " has similarity " + str(document_similarity) + " to IoT Data ")

    # sort the similarity with the highest value first
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # print(list(enumerate(sims)))
    logger.info("printing sorted list ...")
    for pos, (doc_position, doc_score) in list(enumerate(sims)):
        # print(sim)
        # for doc_position, doc_score in sim:
        # pprint(doc_position, doc_score)
        # logger.info(str(doc_score), str(doc_position))
        print("Ontology " + str(ontologies_dict[doc_position]) +
              " has similarity " + str(doc_score) + " to IoT Data ")
    # logger.info(ontologies_dict[doc_position])
    # logger.info("printing indexed ontologies list ...")

    # print(str(len(lsi.print_topics())))
    #print("Printing topcis of LSI model ...")
    # print(lsi.print_topics())
    # print("Size of LSI Model is " + str(len(lsi)))


def toDoAndTest():
    for c in lsi[DIIM_BOW[5:8]]:
        print("Document Topics      : ", c[0])      # [(Topics, Perc Contrib)]
        print("Word id, Topics      : ", c[1][:3])  # [(Word id, [Topics])]
        # [(Word id, [(Topic, Phi Value)])]
        print("Phi Values (word id) : ", c[2][:2])
        print("Word, Topics         : ", [(DIIM_DICT[wd], topic)
              for wd, topic in c[1][:2]])   # [(Word, [Topics])]
        # [(Word, [(Topic, Phi Value)])]
        print("Phi Values (word)    : ", [
              (DIIM_DICT[wd], topic) for wd, topic in c[2][:2]])
        print("------------------------------------------------------\n")


def run_kavita_ex():
    model = gensim.models.Word2Vec(
        DIIM_corpus_processed,
        min_count=1)
    model.train(DIIM_corpus_processed, total_examples=len(
        DIIM_corpus_processed), epochs=10)
    w1 = "sensor"
    logger.info("found similar words for " + w1)
    logger.info(model.wv.most_similar(positive=w1))


def create_corpus(dir):
    document = ""
    for root, dirs, files in os.walk(dir):
        # for file in filter(lambda file: file.endswith('.txt'), files):
        for file in filter(lambda file: file.endswith('.owl') or file.endswith('.txt'), files):
            # for file in filter(lambda file: file.endswith('OGC_SensorML21.txt'), files):
            # read the entire document, as one big string
            document_path = Path(os.path.join(root, file))
            document = (open(document_path).read()).join(document)

    return document


def run_BCU_NLP_Pipeline_ex():
    # train using skip-gram
    skip_gram = True
    # create vocabulary
    logger.info('building vocabulary...')
    model = models.Word2Vec()

    # sentences = DIIM_corpus_processed
    # model.build_vocab(DIIM_corpus_processed)
    # text_corpus = create_corpus(ONTOS_DIR + '\\SensorML')
    # print('len ' + text_corpus)
    # sentences = models.word2vec.LineSentence(text_corpus)

    input_filename = "G:\\MY\\bcu\\OneDrive - Birmingham City University\\phd\\dev\\0Projects\\diim\\DIIM\\res\\data\\input\\ontos\\SensorML\\OGC_SensorML21.txt"
    sentences = models.word2vec.LineSentence(input_filename)

    model.build_vocab(sentences)
    # train model
    logger.info('training model...')
    if skip_gram:
        model.train(sentences, total_examples=model.corpus_count, epochs=3)
    else:
        model.train(sentences, total_examples=model.corpus_count,
                    epochs=model.iter)
    # and save the trained model
    # logger.info('- saving model...')
    # model.save("BCU_NLP_Pipeline_ex_model.tmp")

    # find similar words
    w1 = "sensor"
    print("found similar words for " + w1)
    print(model.wv.most_similar(w1, topn=10))


def old():
    for document in DIIM_corpus_raw:
        document = preprocess_DIIM_document_string(document)
        # Create a set of frequent words
        # Lowercase each document, split it by white space and filter out stopwords
        # print(document)
        words_list = [word for word in document.lower().split()
                      if word not in stoplist]
        # print(len(words_list))
        # print(words_list)
        # words_list_without_slash = []
        # for word in words_list:
        #     words_wo_s = word.split('/')
        #    for word_wo_s in words_wo_s:
        #       if word_wo_s not in stoplist:
        #          words_list_without_slash.append(word_wo_s)
        # print(len(words_list_without_slash))
        # Count word frequencies

        # Only keep words that appear more than once
        words_list = keep_word_with_frequency(words_list, 1)
        # print(len(words_list))
        # pprint.pprint(words_list)


def setup_DIIM_corpus():

    ###################
    # if DIIM CORPUS or DIIM DICT can not be loaded train DIIM Model --> initialize DIIM DICT and CORPUS
    if not(load_DIIM_data()):
        logger.info("Loading of DIIM corpus data failed!")
        # initialize_DIIM_Corpus(GOIoTP_DIR)
        initialize_DIIM_corpus_raw()
        create_DIIM_corpus_processed()
        create_DIIM_DICT()
        create_DIIM_BoW_Vec()

        # save all DIIM corpus related data for next start-up
        store_DIIM_data()
    else:
        logger.info("Loading of DIIM corpus data from storage scuessful!")
    ###############################################################################
    # Before proceeding, we want to associate each word in the corpus with a unique
    # integer ID. We can do this using the :py:class:`gensim.corpora.Dictionary`
    # class.  This dictionary defines the vocabulary of all words that our
    # processing knows about.


def initialize_IoT_dataset():
    logger.info("Initializing IoT datasets from " + IOT_DATASET_PATH)
    global IoT_dataset
    for root, dirs, files in os.walk(IOT_DATASET_PATH):
        # for file in filter(lambda file: file.endswith('.txt'), files):
        for file in filter(lambda file: file.endswith('.csv') or file.endswith('.json'), files):
            if not (file.endswith('.small.json')):  # do not process files ending with .small.json
                # read the entire document, as one big string
                document_path = Path(os.path.join(root, file))
                document = open(document_path).read()

                logger.info(
                    "Adding " + file + " as key to IoT datasets with document word length " + str(len(document)))
                IoT_dataset[file] = document

    logger.info("IoT_dataset initilized with size " + str(len(IoT_dataset)))
    return IoT_dataset


def run_DIIM_routine():
    setup_DIIM_corpus()
    create_vector_space(iot_data)
    run_gensim_coreconcepts_ex()
    #
    # if not load_DIIM_data():
    #    logger.info("Could not load stored DIIM_DATA from " + DIIM_STORE_PATH)
    # initialize_DIIM_corpus_raw()
    # print("ontos dict size " + str(len(ontologies_dict)))
    # print("DIIM corpus size " + str(len(DIIM_corpus_raw)))
    # create_DIIM_corpus_processed()
    # create_DIIM_DICT()
    # create_DIIM_BoW_Vec()


def calculate_similarity_of_IoT_dataset():
    similarity_data = {'Ontologies': ontologies_dict.values()}
    for key in IoT_dataset.keys():
        similarity_data.update(
            calculate_similarity_to_DIIM_Corpus(key, IoT_dataset[key]))
    similarity_dataframe = pd.DataFrame(
        similarity_data)
    return similarity_dataframe


def calculate_similarity_to_DIIM_Corpus(iot_data_name, iot_data):
    print("*** DIIM_BoW_Corpus")
    print(type(DIIM_BoW_corpus))
    print(str(len(DIIM_BoW_corpus)))
    # print(DIIM_BoW_corpus)
    print("*** DIIM_DICT")
    print(type(DIIM_DICT))
    print(str(len(DIIM_DICT)))

    LSI_model = models.LsiModel(DIIM_BoW_corpus, id2word=DIIM_DICT,)
    IoT_BoW_model = create_IoT_BoW_model(iot_data)
    # convert the query to LSI space
    IoT_LSI_vec = LSI_model[IoT_BoW_model]
    logger.info(IoT_LSI_vec)
    # transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(LSI_model[DIIM_BoW_corpus])
    # perform a similarity query against the corpus
    sims = index[IoT_LSI_vec]
    similarity_series = {iot_data_name: sims}
    print(sims)
    # sort the similarity with the highest value first
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # print(list(enumerate(sims)))
    logger.info(
        "printing sorted list of similarities for IoT data " + iot_data_name)
    logger.info('ontologies_dict type', type(ontologies_dict))
    logger.info('Ontologies Dict items', str(ontologies_dict.items))
    logger.info("Ontologies Dict  Keys ", str(ontologies_dict.keys))
    for pos, (doc_position, doc_score) in list(enumerate(sims)):
        print(pos, doc_position, doc_score)
        print("Ontology ", ontologies_dict[doc_position],
              " has similarity ", doc_score, " to IoT Data source ", iot_data_name)
    print('')
    return similarity_series


def show_IoT_similarity_plot(IoT_similarity_values):
    # print(IoT_similarity_values.head())
    #   Ontologies  drinking-water-quality-monitoring.json  IndianRiver-WaterQuality.json  KaaIoTData.json
    # 0      COSMO                                0.365361                       0.082135         0.019622
    # 1      DOLCE                                0.144990                       0.058737         0.033905
    # 2         GO                                0.106100                       0.016020         0.000024
    # 3     GOIoTP                                0.305957                       0.062046         0.012496
    # 4    INSPIRE                                0.020395                       0.018035         0.017274
    IoT_similarity_values.plot(
        kind='barh', x='Ontologies', y='KaaIoTData.json', title='IoT dataset similarity to ontologies and standards')
    plt.show()

    IoT_similarity_values.plot(
        kind='barh', x='Ontologies', title='IoT dataset similarity to ontologies and standards')
    plt.show()


def show_IoT_similarity_plot_with_barlabels():
    # print(IoT_similarity_values.head())
    #   Ontologies  drinking-water-quality-monitoring.json  IndianRiver-WaterQuality.json  KaaIoTData.json
    # 0      COSMO                                0.365361                       0.082135         0.019622
    # 1      DOLCE                                0.144990                       0.058737         0.033905
    # 2         GO                                0.106100                       0.016020         0.000024
    # 3     GOIoTP                                0.305957                       0.062046         0.012496
    # 4    INSPIRE                                0.020395                       0.018035         0.017274
    ontology_names = ontologies_dict.values()
    for key in IoT_dataset.keys():
        similarity_values = calculate_similarity_to_DIIM_Corpus(
            key, IoT_dataset[key])[key]

        # better way to show bars with values
        df = pd.DataFrame(
            {key: similarity_values}, index=ontology_names)
        #df.plot(title='IoT dataset similarity to ontologies and standards')
        ax = df.plot.barh()

        ax.bar_label(ax.containers[0])

        plt.show()


def demo_similarity_IoT_Data_with_Ontologies():
    # Working for DEMO 10/04/2022
    setup_DIIM_corpus()
    # run_DIIM_routine()
    initialize_IoT_dataset()
    IoT_similarity_values = calculate_similarity_of_IoT_dataset()
    show_IoT_similarity_plot(IoT_similarity_values)


def demo_similarity_IoT_Data_with_Ontologies_barlabels():
    setup_DIIM_corpus()
    # run_DIIM_routine()
    initialize_IoT_dataset()
    show_IoT_similarity_plot_with_barlabels()


def main():

    # demo_similarity_IoT_Data_with_Ontologies() # Working for DEMO 10/04/2022
    demo_similarity_IoT_Data_with_Ontologies_barlabels()
    # print(IoT_dataset.keys())
    # calculate_similarity_between_IoT_and_Ontos()
    # use example code to test the word similarity calculation
    # run_kavita_ex()
    # run_BCU_NLP_Pipeline_ex()


main()
