import logging
import DIIM_config as config
import json
import os
import utility as util

logger = logging.getLogger(config.LOGAPPLICATION_NAME)


def serialize_diim_datasets(diim_datasets):
    serialize_diim_datasets_to_JSON(diim_datasets)
    serialize_diim_datasets_to_CSV(diim_datasets)


def serialize_diim_datasets_to_JSON(diim_datasets):
    logger.info('Starting to serialize DIIM Datasets ...')
    for i in diim_datasets:
        fileName = util.get_fileName_without_extension(i.get_name())
        # serialized fileName should start with 'DIIM_Dataset_'
        # and ends with '.json' because it will be saved as json object
        filePath = config.storeDir + 'DIIM_Dataset_' + fileName + '.json'
        # if file already exits in store directory skip serialization
        if os.path.isfile(filePath):
            logger.info('File ' + filePath +
                        ' already exists, therefore skipping serializing file!')
        else:
            logger.info("Encode DIIM Datasets " +
                        str(i.get_id()) + " into JSON formatted Data")
            # diimJSONData = json.dumps(i.toJson())
            diimJSONData = i.toJson()
            logger.info("Writing diim dataset to " + filePath)
            with open(filePath, 'w') as outfile:
                json.dump(diimJSONData, outfile)
            outfile.close
    logger.info('Done with serialization!')


def serialize_diim_datasets_to_CSV(diim_datasets):
    logger.info('Starting to serialize DIIM Datasets ...')
    for i in diim_datasets:
        fileName = util.get_fileName_without_extension(i.get_name())
        # serialized fileName should start with 'DIIM_Dataset_'
        # and ends with '.json' because it will be saved as json object
        filePath = config.storeDir + 'DIIM_Dataset_' + fileName + '.csv'
        # if file already exits in store directory skip serialization
        if os.path.isfile(filePath):
            logger.info('File ' + filePath +
                        ' already exists, therefore skipping serializing file!')
        else:
            logger.info("Encode DIIM Datasets " +
                        str(i.get_id()) + " into JSON formatted Data")
            # diimJSONData = json.dumps(i.toJson())
            #diimJSONData = i.toJson()
            logger.info("Writing diim dataset to " + filePath)
            with open(filePath, 'w') as outfile:
                for key in i.keys():
                    outfile.write("%s,%s\n" % (key, i[key]))
                #json.dump(diimJSONData, outfile)
            outfile.close
    logger.info('Done with serialization!')


def deserialize_diim_datasets(diim_datasets_reloaded):
    '''reads diim datasets from json files '''

    logger.info(
        'Starting to deserialize DIIM Datasets from dir ' + config.storeDir)
    for currentpath, folders, files in os.walk(config.storeDir):
        for file in files:
            filePath = os.path.join(currentpath, file)
            if (file.endswith('.json')):
                logger.info('Reading DIIM dataset from file ' + filePath)
                with open(filePath) as jsonFile:
                    # read object as string
                    diimJSONData = json.load(jsonFile)
                    # convert to dict object again from string representation
                    diimJSONData = json.loads(diimJSONData)
                    # add to list of
                    diim_datasets_reloaded.append(diimJSONData)
    logger.info('Done with deserialization!')


def serialize_diim_similarity_matrix(diim_similarity_matrix):
    # get the name of the stored similarity matrix
    dataset_folder_path, fileName = os.path.split(
        diim_similarity_matrix['filePath'])

    # remove extension of the file
    fileName = util.get_fileName_without_extension(fileName)
    # serialized fileName should start with 'DIIM_Similarity_'
    # and ends with '.json' because it will be saved as json object
    filePath = config.storeDir + 'DIIM_Similarity_' + fileName + '.json'
    # if file already exits in store directory skip serialization
    if os.path.isfile(filePath):
        logger.info('File ' + filePath +
                    ' already exists, therefore skipping serializing file!')
    else:
        logger.info("Encode DIIM similarity matrix " +
                    str(diim_similarity_matrix.get_id()) + " into JSON formatted Data")
        diimJSONData = diim_similarity_matrix.toJson()
        logger.info("Writing diim similarity matrix to " + filePath)
        with open(filePath, 'w') as outfile:
            json.dump(diimJSONData, outfile)
        outfile.close


def serialize_diim_similarity_matrices(diim_similarity_matrices):
    logger.info('Starting to serialize DIIM similarity matrices ...')
    for matrix in diim_similarity_matrices:
        serialize_diim_similarity_matrix(matrix)
    logger.info('Done with serialization!')


def deserialize_diim_similarity_matrix(currentpath, file):
    filePath = os.path.join(currentpath, file)
    if (file.endswith('.json') and file.startswith('DIIM_Similarity')):
        logger.info('Reading DIIM dataset from file ' + filePath)
        with open(filePath) as jsonFile:
            # read object as string
            diimJSONData = json.load(jsonFile)
            # convert to dict object again from string representation
            diimJSONData = json.loads(diimJSONData)
            return diimJSONData
            # add to list of


def deserialize_diim_similarity_matrices(diim_similarity_matrices_reloaded):
    '''reads diim datasets from json files '''
    logger.info(
        'Starting to deserialize DIIM Similarity matrices from dir ' + config.storeDir)
    for currentpath, folders, files in os.walk(config.storeDir):
        for file in files:
            diimJSONData = deserialize_diim_similarity_matrix(
                currentpath, file)
            diim_similarity_matrices_reloaded.append(diimJSONData)
    logger.info('Done with deserialization!')
