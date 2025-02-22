# TODO: Use pckg_resources to load files, when distributed as package
import pandas as pd
import numpy as np
import os

def _base_path():
    """Create and return path to package. 
    
    (Temporary hacky solution)
    """
    drive, filedir = os.path.splitdrive(__file__)
    parentdir = filedir.split(os.sep)[:-2]
    parentdir[0] = os.sep
    return os.path.join(drive, *parentdir)

_basepath = _base_path()

resources = os.path.join(_basepath, 'resources')
output = os.path.join(_basepath, 'output')

def load_dataset(name, version):
    """Return specified dataset, ready to be used with package modules.

    Parameters:
        name -- dataset name
        version -- dataset version name
    
    Arguments:
        name:
            'daganlevy' -- Dataset, created by Levy & Dagan (2016)
            'daganlevy_lemmatised' -- Lemmatised version of Levy & Dagan
            'zeichner' -- Dataset, creted by Zeichnner et al. (2012)
        
        version:
            'original' -- Original dataset as pandas.core.frame.DataFrame
            'tidy' -- Dataset for comparison as pandas.core.frame.DataFrame
            'analysis' -- Dataset for inference classifier as numpy.ndarray (2,3)
            'lemmatised' -- Dataset for comparison as pandas.core.frame.DataFrame. Only available for 'daganlevy'!
    
    Returns:
        dataset ,'original'/'tidy' -- pandas.core.frame.DataFrame
        dataset ,'analysis' -- numpy.ndarray (2,3)
    """
    datasets = ['daganlevy', 'daganlevy_lemmatised', 'zeichner']
    versions = ['original', 'tidy', 'analysis', 'lemmatised']
    
    if name not in datasets:
        raise ValueError('"' + name + '" is not a valid dataset name. Valid names: ' + ', '.join(datasets))
    
    if version not in versions:
        raise ValueError('"' + version + '" is not a valid version name. Valid names: ' + ', '.join(versions))
    
    if version == 'original':
        filepath = os.path.join(resources, 'original-datasets', name + '.txt')
        return pd.read_csv(filepath)
    
    if version == 'tidy':
        filepath = os.path.join(resources, 'datasets', name + '-tidy.csv')
        return pd.read_csv(filepath, index_col=0)
    
    if version == 'analysis':
        filepath = os.path.join(resources, 'datasets', name + '.npy')
        return np.load(filepath)
    
def load_result(dataset):
    datasets = ['daganlevy', 'daganlevy_lemmatised', 'zeichner']
    
    if dataset not in datasets:
        raise ValueError('{0} is not a valid dataset name. Valid names: {1}'.format(dataset, datasets))
    else:
        filepath = os.path.join(output, dataset + '_result.csv')
        return pd.read_csv(filepath, index_col=0)

def load_resource(module, res):
    """Return specified resource.

    Parameters:
        module -- Name of the module, requiring the resource
        res -- Name of the resource
    
    Arguments:
        'EntailmentGraph':
            'berant_2011' -- Edgelist of Berant (2011) Entailment Graph, specifying the graph as numpy.ndarray (2,)
            'berant_2011_no-context' -- Edgelist of Berant (2011) Entailment Graph, without context representation specifying the graph as numpy.ndarray (2,)
            'typemap' -- Maping of argument instances (keys) to types (values) as dict
            'lambda=0.1' -- Edgelist for Berant et al. (2010) Entailment Graph, as numpy.ndarray (2,)
            'lambda=0.05' -- Edgelist for Berant et al. (2010) Entailment Graph, as numpy.ndarray (2,)

        'PPDB2':
            'db' -- Path to PPDB2.0 XXXXL sqlite file, for PPDB2 Classifier
            'db-mini' -- Path to PPDB2.0 XS sqlite file, for PPDB2 Classifier
    
    Returns:
        "EntailmentGraph", ['edgelist', 'lambda=0.1', 'lambda=0.05'] -- numpy.ndarray (2,)
        "EntailmentGraph", 'typemap' -- dict
        "PPDB2", * -- string 
    
    Raises:
        ValueError -- If incorrect module name was provided
    """
    modules = ['EntailmentGraph', 'PPDB2']
    if module not in modules:
        raise ValueError('"' + module + '" is not a valid module name. Valid names:' + modules)
    
    filepath = os.path.join(resources, module)
    
    if module == 'EntailmentGraph':
        if res == 'edgelist':
            filepath = os.path.join(filepath,  'entailment-graph.json')
            return pd.read_json(filepath).reindex(columns=['text', 'hypothesis']).reset_index(drop=True)[['text','hypothesis']].values

        if res == 'berant_2011':
            filepath = os.path.join(filepath, 'berant_2011_typed.npy')
            return np.load(filepath)

        if res == 'berant_2011_no-context':
            filepath = os.path.join(filepath, 'berant_2011_mapped.npy')
            return np.load(filepath)
        
        if res == 'lambda=0.1':
            filepath = os.path.join(filepath, 'berant_2010-0.1.npy')
            return np.load(filepath)

        if res == 'lambda=0.1,mapped':
            filepath = os.path.join(filepath, 'berant_2010-0.1_mapped.npy')
            return np.load(filepath)

        if res == 'lambda=0.05':
            filepath = os.path.join(filepath, 'berant_2010-0.05.npy')
            return np.load(filepath)

        if res == 'lambda=0.05,mapped':
            filepath = os.path.join(filepath, 'berant_2010-0.05_mapped.npy')
            return np.load(filepath)

        if res == 'typemap':
            filepath = os.path.join(filepath, 'class-instance-mapping.txt')
            data = pd.read_csv(filepath, sep='\t').iloc[:,[1,0]].values
            return {instance: type for instance, type in data}

    if module == 'PPDB2':    
        if res == 'db':
            filepath = os.path.join(filepath, 'ppdb2.sqlite')
            return filepath
        
        if res == 'db-mini':
            filepath = os.path.join(filepath, 'ppdb.sqlite')
            return filepath

        if res == 'rules':
            filepath = os.path.join(filepath, 'ppdb2.csv')
            return pd.read_csv(filepath).values