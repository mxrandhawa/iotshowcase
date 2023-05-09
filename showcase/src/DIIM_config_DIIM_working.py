import os
import logging

# File paths
# define your project dir here
#projectDir = "G:/MY/bcu/OneDrive - Birmingham City University/phd/dev/0Projects/graph4nlp/examples/diim/"
# let program dedect the project dir

# TODO: remove or rename to capital constants
# projectDir = os.getcwd() + "/"

# inputDir = resDir + "input/"
# outputDir = resDir + "output/"
# storeDir = resDir + "store/"


PROJECT_DIR = os.getcwd() + "/"
RES_DIR = PROJECT_DIR + "res/data/"
ONTOS_DIR = PROJECT_DIR + 'res/data/input/ontos'
IOT_DIR = PROJECT_DIR + 'res/data/input/iot4w2v'

DIIM_STORE_PATH = PROJECT_DIR + 'res/data/store'

DIIM_CORPORA_STORE_PATH = DIIM_STORE_PATH + '/corpora'
DIIM_ONTO_CORPORA_STORE_PATH = DIIM_CORPORA_STORE_PATH + '/onto'
DIIM_IOT_CORPORA_STORE_PATH = DIIM_CORPORA_STORE_PATH + '/iot'

DIIM_LSI_STORE_PATH = PROJECT_DIR + 'res/data/store/lsi'
DIIM_LSI_ONTO_STORE_PATH = DIIM_LSI_STORE_PATH + '/onto'
DIIM_LSI_IOT_STORE_PATH = DIIM_LSI_STORE_PATH + '/iot'

DIIM_DICT_STORE_PATH = DIIM_STORE_PATH + '/dict'
DIIM_DICT_ONTO_STORE_PATH = DIIM_DICT_STORE_PATH + '/onto'
DIIM_DICT_IOT_STORE_PATH = DIIM_DICT_STORE_PATH + '/iot'

DIIM_DF_STORE_PATH = DIIM_STORE_PATH + '/df'
DIIM_WV_STORE_PATH = DIIM_STORE_PATH + '/wv'
# similarities of iot with ontos
DIIM_SIM_STORE_PATH = DIIM_STORE_PATH + '/sim'

DIIM_IMG_STORE_PATH = DIIM_STORE_PATH + '/img'

DIIM_TMP_DIR_PATH = RES_DIR + "tmp/"



LOGFILE_NAME = 'diim.log'
LOGAPPLICATION_NAME = 'diim_application'

# Test data for small test programs
# it could be deleted later
inputURL = 'file:///G:/MY/bcu/OneDrive - Birmingham City University/phd/dev/0Projects/graph4nlp/examples/diim/res/data/input/json/KaaIoTData.json'
datasetFilename = 'WikiDataset.txt'
cleanedDatasetFilename = 'WikiDatasetCleaned.txt'
posTaggedWikiDatasetFileName = 'WikiDataset_POS.txt'
modelFilename = 'WikiDatasetCleaned.model'
word = "water"

'''
def initialize():
    inputURL = 'https://en.wikipedia.org/wiki/Biomedicine'
    outputFilename = 'WikiDataset.txt'
    projectDir = "G:/MY/bcu/OneDrive - Birmingham City University/phd/dev/0Projects/graph4nlp/examples/diim/src/ex/NLP-Word2Vec"
'''

'''
def getProjectDir():
    if projectDir == "":
        initialize()
    return projectDir
'''


# def setupLogger():
# create logger with 'dimm_application'
logger = logging.getLogger(LOGAPPLICATION_NAME)
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(LOGFILE_NAME)
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# formatter = logging.Formatter('%(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# logger.info('creating an instance of AnalyseJSON')
# a = ajd.Auxiliary()
# logger.info('created an instance of AnalyseJSON')
