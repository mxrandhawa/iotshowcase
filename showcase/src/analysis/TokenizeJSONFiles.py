from os import add_dll_directory
from nltk.stem import WordNetLemmatizer, wordnet
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from itertools import chain, starmap

import json
import sys
import nltk
# nltk.download()

file1 = "KaaIoTData.json"  # string: name of file
# string: name of file
file2 = "drinking-water-quality-distribution-monitoring-data-csv_json.json"
file3 = "IndianRiver-WaterQuality.json"

JSONFiles = {file1, file2, file3}


def extractKeysOfJSONFile(JSONFileName):
    # original_stdout = sys.stdout  # Save a reference to the original standard output
    # Change the standard output to the file we created.
    # sys.stdout = OutputFile
    global outputString
    global out

    outputString += "Started reading JSON file "
    outputString += JSONFileName
    outputString += "\n"

    with open(JSONFileName, "r") as read_file:
        outputString += "Converting JSON encoded data into Python dictionary \n"
        jsonData = json.load(read_file)
        wordTokens = set()

    outputString += "Decoded JSON Data From File \n"
    keys = set()
    outputString += "Extract keys from JSON data \n"
    jsonData = flatten_json(jsonData)
    # df = flatten_json_iterative_solution(jsonData)
    for key in jsonData:
        # print(key)
        extractedKeys = key.split("_")
        for eKey in extractedKeys:
            str(eKey)
            if (eKey.isnumeric() != True):
                keys.add(eKey)

    print(keys)
    outputString += str(keys)
    outputString += "\n"
    outputString += "Done reading json file \n\n"
    read_file.close()
    return keys


def flatten_json_iterative_solution(dictionary):
    """Flatten a nested json file"""

    def unpack(parent_key, parent_value):
        """Unpack one level of nesting in json file"""
        # Unpack one level only!!!

        if isinstance(parent_value, dict):
            for key, value in parent_value.items():
                temp1 = parent_key + '_' + key
                yield temp1, value
        elif isinstance(parent_value, list):
            i = 0
            for value in parent_value:
                temp2 = parent_key + '_'+str(i)
                i += 1
                yield temp2, value
        else:
            yield parent_key, parent_value

    # Keep iterating until the termination condition is satisfied
    while True:
        # Keep unpacking the json file until all values are atomic elements (not dictionary or list)
        dictionary = dict(chain.from_iterable(
            starmap(unpack, dictionary.items())))
        # Terminate condition: not any value in the json file is dictionary or list
        if not any(isinstance(value, dict) for value in dictionary.values()) and \
           not any(isinstance(value, list) for value in dictionary.values()):
            break

    return dictionary

# !!not working with fattered JSON object


def parse_json_recursively(jsonData, keys):
    for json_object in jsonData:
        print(type(json_object))
        if(type(json_object) == str):
            keys.add(str(json_object))
        elif type(json_object) is dict and json_object:
            for key in json_object:
                if(type(key) == str):
                    keys.add(str(key))
                parse_json_recursively(json_object[key], keys)
        elif type(json_object) is list and json_object:
            for item in json_object:
                parse_json_recursively(item, keys)


def synonymsOfKeys(keys):
    # outputString = "" # reset the string for lemmmas
    global outputString
    for key in keys:
        outputString += "\nFind Synonyms of *"
        outputString += str(key)
        outputString += "* \n"
        synsets = wordnet.synsets(key)
        if len(synsets) == 0:
            outputString += "No synonyms found \n"
        for syn in synsets:
            outputString += str(syn)
            outputString += "\n"
            outputString += "Find lemmas of *"
            outputString += str(syn)
            outputString += "* \n"
            lemmas = syn.lemmas()
            if len(lemmas) == 0:
                outputString += "No lemmas found \n"
            for lemma in lemmas:
                outputString += lemma.name()
                outputString += " "
            outputString += " \n"


def flatten_json(nested_json):
    """
        Flatten json object with nested keys into a single level.
        Args:
            nested_json: A nested json object.
        Returns:
            The flattened json object if successful, None otherwise.
    """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out


def analyzeJSONFile(file):
    global outputString
    outputString = ""
    OutputFile = open("Out-" + file, 'w')
    # Input: arg1 = name of the JSON file, arg2 = write output in *outputString
    keys = extractKeysOfJSONFile(file)
    # convert set to list, sort it and write in file
    keysList = list(keys)
    keysList.sort()
    outputString += "Sorted list of keywords: \n"
    outputString += str(keysList)
    outputString += "\n"
    # find synonymsOfKeys
    synonymsOfKeys(keysList)
    # Writing and closing output file
    OutputFile.write(str(outputString))
    OutputFile.close()


for file in JSONFiles:
    analyzeJSONFile(file)
