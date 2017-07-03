import pandas as pd
import os

# TODO: Use pckg_resources to load files, when distributed as package

# Creates the path to the resources, independent of calling frame.
# Don't even ask...
def makeBasePath():
    drive, filedir = os.path.splitdrive(__file__)
    parentdir = filedir.split(os.sep)[:-2]
    parentdir[0] = '\\'
    return os.path.join(drive, *parentdir)
    return os.path.join(*parentdir)

basepath = makeBasePath()
resources = os.path.join(basepath, 'resources')
output = os.path.join(basepath, 'output')

#TODO: create /original-datasets/zeichner.txt
def load(name, version):
    if (name != 'daganlevy') and (name != 'zeichner'):
        raise ValueError('"' + name + '" is not a valid dataset name. Valid names: "daganlevy", "zeichner"')
    
    if version == 'original':
        filepath = os.path.join(resources, 'original-datasets', name + '.txt')
        return pd.read_csv(filepath)
    
    if version == 'tidy':
        filepath = os.path.join(resources, 'datasets', name + '-tidy.csv')
        return pd.read_csv(filepath) 
    
    if version == 'analysis':
        filepath = os.path.join(resources, 'datasets', name + '.json')
        return pd.read_json(filepath).reindex(columns=['text', 'hypothesis', 'entailment']).reset_index(drop=True)
    
    raise ValueError('"' + version + '" is not a valid version name. Valid names: "original", "tidy", "analysis"')


#import sys
#sys.path.append('C:\\Users\\Nev\\Projects\\bachelor_thesis')
#import utils.io
#utils.io.load('daganlevy', 'tidy')