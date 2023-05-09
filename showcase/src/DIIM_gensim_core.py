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
import pprint
import re
import json
import logging
from gensim import corpora
from gensim import similarities
from numpy import number

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


projectDir = os.getcwd() + "\\"
indexFilepath = projectDir + 'res\\\data\\tmp\\diim_ontos.index'
GOIoTP_DIR = projectDir + 'res\\data\\input\\ontos\\domain\\GOIoTP'
ONTOS_DIR = projectDir + 'res\\data\\input\\ontos'
IoT_Data_DIR = projectDir + 'res\\data\\input\\json'
DIIM_STORE_PATH = projectDir + 'res\\data\\store'
# path to save the DIIM Corpus
DIIM_CORPUS_PATH = projectDir + 'res\\data\\store\\DIIM_CORPUS'
# path to save the DIIM dictionary
DIIM_DICT_PATH = projectDir + 'res\\data\\store\\DIIM_DICT'
DIIM_DICT = corpora.Dictionary('')

# GOIoTP_D1 = ''
ontologies_dict = {}  # set of ontologies
ontologies_dict_counter = 0  # counter for ontologies
iot_data_dict = {}  # set of iot data documents, which will be used to find similarity
iot_data_dict_counter = 0  # counter for iot data documents
DIIM_CORPUS_RAW = []  # processed model from ontologies
DIIM_CORPUS_PROCESSED = []  # processed model from ontologies
DIIM_Text_Corpus = []  # list of documents to train the model, that will be ontologies
DIIM_Corpus_for_training = []
DIIM_BOW_Corpus = []
DIIM_BOW = []  # DIIM represented as a Bag of Words vector
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
iot_data = "Meta data: Name:S1, Description: The sensor measures water temperature inFahrenheit, serial:00-14-22-01-23-45, model: BFG9000, mac:50:8c:b1:77:e8:e6, latitude:51.75543, longitude:-1.03248 Measurement data:4baa-a2ff-8741efad4e63: temp:timestamp:2021-08-09T17:01:28.796Z, values: value :20, timestamp:2021-08-09T17:01:38.792Z, values:value:24"
IoT_Data_BoW = []


def initialize_DIIM_CORPUS_RAW():
    logger.info("Initializing DIIM_CORPUS_RAW from " + ONTOS_DIR)
    ontologies_dict_counter = 0  # counter for ontologies
    number_of_read_files = 0
    for root, dirs, files in os.walk(ONTOS_DIR):
        document = ""
        document_dir_name = ''
        # for file in filter(lambda file: file.endswith('.txt'), files):
        for file in filter(lambda file: file.endswith('.owl') or file.endswith('.txt'), files):
            # for file in filter(lambda file: file.endswith('OGC_SensorML21.txt'), files):
            # read the entire document, as one big string
            document_path = Path(os.path.join(root, file))
            document += open(document_path).read()
            number_of_read_files += 1
            if (len(document_dir_name) == 0):
                # get then name of directory
                document_dir_names = str(document_path.parent).split("\\")
                document_dir_name = document_dir_names[len(
                    document_dir_names)-1]
            logger.info("Adding " + file + " from Dir " +
                        document_dir_name + " to DIIM_CORPUS_RAW")

        if (len(document_dir_name) > 0 and len(document) > 0):
            global ontologies_dict
            # add name of directory as ontology name to ontologies dictionary
            ontologies_dict[ontologies_dict_counter] = document_dir_name
            logger.info("Adding " + document_dir_name + " with key " +
                        str(ontologies_dict_counter) + " to ontologies dictionary")
            ontologies_dict_counter = 1 + ontologies_dict_counter
        # append aggregated text of all files of directory (ontology) to the corpus
        DIIM_CORPUS_RAW.append(document)
        logger.info("DIIM_CORPUS_RAW initialized with " +
                    str(len(DIIM_CORPUS_RAW)) + " documents")
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


def preprocess_DIIM_CORPUS_RAW():
    logger.info("Preprocessing DIIM_CORPUS_RAW ... ")
    logger.info("Preprocessed DIIM_CORPUS has length " +
                str(len(DIIM_CORPUS_RAW)))
    # Lowercase each document, split it by white space and filter out stopwords
    preprocessed_diim_documents = [preprocess_DIIM_document_string(document)
                                   for document in DIIM_CORPUS_RAW]
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
    logger.info("Preprocessing of DIIM_CORPUS_RAW done !")
    logger.info("Preprocessed DIIM_CORPUS has length " +
                str(len(processed_corpus)))
    return processed_corpus


def create_DIIM_CORPUS_PROCESSED():
    global DIIM_CORPUS_PROCESSED
    DIIM_CORPUS_PROCESSED = preprocess_DIIM_CORPUS_RAW()
    return DIIM_CORPUS_PROCESSED


def create_DIIM_DICT():
    global DIIM_DICT
    DIIM_DICT = corpora.Dictionary(DIIM_CORPUS_PROCESSED)
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
    # pprint.pprint(DIIM_CORPUS_PROCESSED)
    return filterd_list


def store_DIIM_CORPUS_PROCESSED():
    global DIIM_CORPUS_PROCESSED
#    print(len(DIIM_CORPUS_PROCESSED))
 #   print(len(ontologies_dict))
    encoded_diim_corpus_processed = json.dumps(DIIM_CORPUS_PROCESSED)
    jsonFile = open(DIIM_STORE_PATH + "\\DIIM_CORPUS_PROCESSED.json", "w")
    jsonFile.write(encoded_diim_corpus_processed)
    jsonFile.close()


def load_DIIM_CORPUS_PROCESSED():
    logger.info(
        'Trying to load DIIM_CORPUS_PROCESSED from ' + DIIM_STORE_PATH)
    fileObject = open(DIIM_STORE_PATH + "\\DIIM_CORPUS_PROCESSED.json", "r")
    jsonContent = fileObject.read()
    global DIIM_CORPUS_PROCESSED
    DIIM_CORPUS_PROCESSED = json.loads(jsonContent)
    return True


def store_DIIM_CORPUS_RAW():
    global DIIM_CORPUS_RAW
    encoded_diim_corpus_raw = json.dumps(DIIM_CORPUS_RAW)
    jsonFile = open(DIIM_STORE_PATH + "\\DIIM_CORPUS_RAW.json", "w")
    jsonFile.write(encoded_diim_corpus_raw)
    jsonFile.close()


def load_DIIM_CORPUS_RAW():
    logger.info('Trying to load DIIM_CORPUS_RAW from ' + DIIM_STORE_PATH)
    fileObject = open(DIIM_STORE_PATH + "\\DIIM_CORPUS_RAW.json", "r")
    jsonContent = fileObject.read()
    global DIIM_CORPUS_RAW
    DIIM_CORPUS_RAW = json.loads(jsonContent)
    return True


def store_DIIM_DICT():
    DIIM_DICT.save(DIIM_STORE_PATH + "\\DIIM_DICT.json")


def load_DIIM_DICT():
    global DIIM_DICT
    DIIM_DICT = corpora.Dictionary.load(DIIM_STORE_PATH + "\\DIIM_DICT.json")
    return True


def store_DIIM_data():
    store_DIIM_CORPUS_RAW()
    store_DIIM_CORPUS_PROCESSED()
    store_DIIM_DICT()


def load_DIIM_data():
    logger.info('Trying to load DIIM DATA from ' + DIIM_STORE_PATH)
    # loads the data
    load_DIIM_CORPUS_RAW()
    load_DIIM_CORPUS_PROCESSED()
    load_DIIM_DICT()


def create_vector_space(iot_data):
    iot_data_processed = preprocess_DIIM_document_string(iot_data)
    # Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.
    global IoT_Data_BoW
    IoT_Data_BoW = DIIM_DICT.doc2bow(iot_data_processed.lower().split())
    for token_id, token_count in IoT_Data_BoW:
        logger.info("Token \'" + DIIM_DICT[token_id] +
                    "\' occured " + str(token_count) + ' time(s) in IoT Data')
    return IoT_Data_BoW


def create_DIIM_BoW_Vec():
    global DIIM_BOW
    # print(len(DIIM_CORPUS_PROCESSED))
    # print(DIIM_DICT)
    DIIM_BOW = [DIIM_DICT.doc2bow(text) for text in DIIM_CORPUS_PROCESSED]
    return DIIM_BOW


def run_gensim_coreconcepts_ex():
    lsi = models.LsiModel(DIIM_BOW, id2word=DIIM_DICT)
    vec_lsi = lsi[IoT_Data_BoW]  # convert the query to LSI space
    logger.info(vec_lsi)
    # transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(lsi[DIIM_BOW])
    indexFilepath
    index.save(indexFilepath)
    index = similarities.MatrixSimilarity.load(indexFilepath)
    # print(create_DIIM_BoW_Vec())
    # store_DIIM_data()
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    # print (document_number, document_similarity) 2-tuples
    for document_number, document_similarity in enumerate(sims):
        logger.info("Ontology " + str(document_number) +
                    " has similarity " + str(document_similarity) + " to IoT Data ")

    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(list(enumerate(sims)))
    logger.info("printing sorted list ...")
    # for doc_position, doc_score in sims:
    #pprint(doc_position, doc_score)
    #logger.info(str(doc_score), str(doc_position))
    # print("Ontology " + ontologies_dict(doc_position) +
    #     " has similarity" + str(doc_score) + "to IoT Data ")
    logger.info("printing indexed ontologies list ...")
    logger.info(ontologies_dict)
    # print(str(len(lsi.print_topics())))
    # print(lsi.print_topics())
    # pprint(lsi.print_topics())


def extra():
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
        DIIM_CORPUS_PROCESSED,
        min_count=1)
    model.train(DIIM_CORPUS_PROCESSED, total_examples=len(
        DIIM_CORPUS_PROCESSED), epochs=10)
    w1 = "sensor"
    logger.info("found similar words for " + w1)
    logger.info(model.wv.most_similar(positive=w1))


def run_BCU_NLP_Pipeline_ex():
    # train using skip-gram
    skip_gram = True
    # create vocabulary
    logger.info('building vocabulary...')
    model = models.Word2Vec()
    # sentences = models.word2vec.LineSentence(input_filename)

    sentences = DIIM_CORPUS_PROCESSED
    model.build_vocab(DIIM_CORPUS_PROCESSED)
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
    logger.info("found similar words for " + w1)
    logger.info(model.wv.most_similar(w1, topn=10))


def old():
    for document in DIIM_CORPUS_RAW:
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


def main():
    # if not load_DIIM_data():
    #    logger.info("Could not load stored DIIM_DATA from " + DIIM_STORE_PATH)
    initialize_DIIM_CORPUS_RAW()
    create_DIIM_CORPUS_PROCESSED()
    create_DIIM_DICT()
    create_DIIM_BoW_Vec()
    create_vector_space(iot_data)
    # use example code to test the word similarity calculation
    # run_kavita_ex()
    # run_BCU_NLP_Pipeline_ex()
    run_gensim_coreconcepts_ex()


main()
