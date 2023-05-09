import os
import shutil

import DIIM_config as config



def make_dir(path):
    logger = config.logger
    # Check whether the specified
    # path exists or not
    isExist = os.path.exists(path)
    if not isExist :
        try: 
            logger.info('Does not exists: ' + path)
            os.mkdir(path) 
            logger.info('Created: ' + path)
        except OSError as error: 
            print(error)
    logger.info('Exists: ' + path)
   
def init_dirs():
    logger = config.logger
    # Specify path for store
    path = config.DIIM_OUTPUT_DIR
    make_dir(path)

    # Specify path for store
    path = config.DIIM_STORE_PATH
    make_dir(path)

    # Corpora paths
    # Specify path 
    path = config.DIIM_CORPORA_STORE_PATH
    make_dir(path)
    # Specify path 
    path = config.DIIM_ONTO_CORPORA_STORE_PATH
    make_dir(path)
    # Specify path 
    path = config.DIIM_IOT_CORPORA_STORE_PATH
    make_dir(path)

    # LSI paths
    # Specify path 
    path = config.DIIM_LSI_STORE_PATH
    make_dir(path)
    # Specify path 
    path = config.DIIM_LSI_ONTO_STORE_PATH
    make_dir(path)
    # Specify path 
    path = config.DIIM_LSI_IOT_STORE_PATH
    make_dir(path)

    # Dictionary paths
    # Specify path 
    path = config.DIIM_DICT_STORE_PATH
    make_dir(path)
    # Specify path 
    path = config.DIIM_DICT_ONTO_STORE_PATH
    make_dir(path)
    # Specify path 
    path = config.DIIM_DICT_IOT_STORE_PATH
    make_dir(path)

    # Dataframe paths
    # Specify path 
    path = config.DIIM_DF_STORE_PATH
    make_dir(path)
    # Specify path 
    path = config.DIIM_WV_STORE_PATH
    make_dir(path)
 

    # Specify Similarities path 
    path = config.DIIM_SIM_STORE_PATH
    make_dir(path)

    # Specify Knowledge Graph Similarities path 
    path = config.DIIM_SIM_KG_STORE_PATH
    make_dir(path)

    # Specify imge storage path 
    path = config.DIIM_IMG_STORE_PATH
    make_dir(path)



    # Specify Temp path 
    path = config.DIIM_TMP_DIR_PATH
    make_dir(path)
 
 
def delete_output_directories():
    logger = config.logger
    dir_path = config.DIIM_OUTPUT_DIR
    try:
        shutil.rmtree(dir_path)
        logger.info('All files deleted in ' + dir_path)
    except OSError as e:
        logger.error("Error: %s : %s" % (dir_path, e.strerror))

def delete_logfile():
    file_path = config.LOGFILE_NAME
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            print("Error: %s : %s" % (file_path, e.strerror))

def diim_init_all():
    delete_logfile()
    delete_output_directories()
    init_dirs()

def diim_init_eco():
    init_dirs()
