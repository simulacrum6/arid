# TODO: Use pckg_resources to load files, when distributed as package
import pandas as pd
import os

def _base_path():
    '''Create and return path to package. 
    
    (Temporary hacky solution)
    '''
    drive, filedir = os.path.splitdrive(__file__)
    parentdir = filedir.split(os.sep)[:-2]
    parentdir[0] = os.sep
    return os.path.join(drive, *parentdir)

_basepath = _base_path()

resources = os.path.join(_basepath, 'resources')
output = os.path.join(_basepath, 'output')

#TODO: create /original-datasets/zeichner.txt
def load_dataset(name, version):
    '''Return specified dataset, ready to be used with package modules.

    Parameters:
        name -- dataset name
        version -- dataset version name
    
    Arguments:
        name:
            "daganlevy" -- Dataset, created by Levy & Dagan (2016)
            "zeichner -- Dataset, creted by Zeichnner et al. (2012)
        
        version:
            "original" -- Original dataset as pandas.core.frame.DataFrame
            "tidy" -- Dataset for comparison as pandas.core.frame.DataFrame
            "analysis" -- Dataset for inference classifier as numpy.ndarray (2,3)
    
    Returns:
        any ,"original"/"tidy" -- pandas.core.frame.DataFrame
        any ,"analysis" -- numpy.ndarray (2,3)
    '''
    if name not in ['daganlevy', 'zeichner']:
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
    '''Return specified resource.

    Parameters:
        module -- Name of the module, requiring the resource
        res -- Name of the resource
    
    Arguments:
        "entailment-graph":
            "edgelist" -- Edgelist, specifying the graph as numpy.ndarray (2,)
            "typemap" -- Maping of argument instances (keys) to types (values) as dict
    
    Returns:
        "entailment-graph", "edgelist" -- numpy.ndarray (2,)
        "entailment-graph", "typemap" -- dict
    
    Raises:
        ValueError -- If incorrect module name was provided
    '''
    if module not in ['entailment-graph', 'ppdb2']:
        raise ValueError('"' + module + '" is not a valid module name. Valid names: "entailment-graph", "ppdb2"')
    
    filepath = os.path.join(resources, module)
    
    if res == 'edgelist':
        filepath = os.path.join(filepath,  'entailment-graph_tidy.csv')
        return pd.read_csv(filepath).values
    
    if res == 'typemap':
        filepath = os.path.join(filepath, 'class-instance-mapping.txt')
        data = pd.read_csv(filepath, sep='\t').iloc[:,[1,0]].values
        return {instance: type for instance, type in data}
    
    if res == 'db':
        filepath = os.path.join(filepath, 'ppdb2.sqlite')
        return filepath