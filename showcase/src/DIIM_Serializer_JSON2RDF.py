from xml.etree.ElementTree import tostring
from xml.etree.ElementTree import Element
import json
import os
import DIIM_config as config

from pip import main


def translate_JSON2RDF():
    for currentpath, folders, files in os.walk(config.inputDir):
        for file in files:
            filePath = os.path.join(currentpath, file)
            if (file.endswith('.json')):
                print('Reading DIIM dataset from file ' + filePath)
                with open(filePath) as jsonFile:
                    # read object as string
                    diimJSONData = json.load(jsonFile)
                    print('DiimData type 1 ' + str(type(diimJSONData)))
                    # convert to dict object again from string representation
                    diimJSONData = json.loads(diimJSONData)
                    print('DiimData type 2 ' + str(type(diimJSONData)))
                    # add to list of
                    return diimJSONData


def dict_to_xml(tag, d):
    '''
    Turn a simple dict of key/value pairs into XML
    '''
    elem = Element(tag)
    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)
    return elem


def main():
    content = translate_JSON2RDF()
    for i in range(1, len(content)):
        e = dict_to_xml('SomeNode', content[i][0])

        # print e
        print(e)


main()
