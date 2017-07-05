import pandas as pd
import os

# TODO: Use pckg_resources to load files, when distributed as package

# Creates the path to the resources, independent of calling frame.
# Don't even ask...
def base_path():
    drive, filedir = os.path.splitdrive(__file__)
    parentdir = filedir.split(os.sep)[:-2]
    parentdir[0] = os.sep
    return os.path.join(drive, *parentdir)
    return os.path.join(*parentdir)

basepath = base_path()
resources = os.path.join(basepath, 'resources')
output = os.path.join(basepath, 'output')

#TODO: create /original-datasets/zeichner.txt
def load_dataset(name, version):
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
        return pd.read_json(filepath).reindex(columns=['text', 'hypothesis', 'entailment']).reset_index(drop=True)[['text','hypothesis']].values
    
    raise ValueError('"' + version + '" is not a valid version name. Valid names: "original", "tidy", "analysis"')

def load_resource(module, res):
    if module not in ['entailment-graph']:
        raise ValueError('"' + module + '" is not a valid module name. Valid names: "entailment-graph"')
    
    filepath = os.path.join(resources, module)
    
    if res == 'edgelist':
        filepath = os.path.join(filepath,  'entailment-graph_tidy.csv')
        return pd.read_csv(filepath).values
    
    if res == 'typemap':
        filepath = os.path.join(filepath, 'class-instance-mapping.txt')
        data = pd.read_csv(filepath, sep='\t').iloc[:,[1,0]].values
        return {instance: type for instance, type in data}

