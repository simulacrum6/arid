import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

def size(dataset):
	return len(dataset)

def positives(dataset):
	return dataset[dataset['entailment'] == True]

def negatives(dataset):
	return dataset[dataset['entailment'] == False]

def uniques(dataset, columns):
    result = []
    for name in columns:
        result.extend(dataset[name])
    return set(result)

def uniquePredicates(dataset):
	return set(dataset['tpred'].append(dataset['hpred']))

# TODO: lexical difference in preds

# TODO: lemmatised daganlevy

def unique_attributes(dataset):
	return set(dataset['ty'].append([
		dataset['tx'], 
		dataset['hx'], 
		dataset['hy']]))

def unique_templates(dataset):
    return set(dataset['text'].append(dataset['hypothesis']))

#TODO: create abstracted coverage(A,B) method
def pred_coverage(datasetA, datasetB):
	predsA = uniquePredicates(datasetA)
	predsB = uniquePredicates(datasetB)
	return len(predsA & predsB) / len(predsB)

def template_coverage(datasetA, datasetB):
    templatesA = unique_templates(datasetA)
    templatesB = unique_templates(datasetB)    
    return len(templatesA & templatesB) / len(templatesB)

def jaccard_index(listA, listB):
    A = set(listA)
    B = set(listB)
    union = A.union(B)
    intersection = A.intersection(B)
    return len(intersection) / len(union)

# FIXME: unique predicate / attribute / template percentages
def dataset_stats(dataset):
	return pd.Series(OrderedDict({
		'size': size(dataset),
		'positives': len(positives(dataset)),
		'negatives': len(negatives(dataset)),
		'pn_rate': len(positives(dataset)) / len(negatives(dataset)),
        'unique_templates': len(unique_templates(dataset)),
		'unique_templates%': len(unique_templates(dataset)) / (size(dataset)*2),
        'unique_templates_T': dataset['text'].nunique(),
        'unique_templates_H': dataset['hypothesis'].nunique(),
        'unique_predicates': len(uniquePredicates(dataset)),
        'unique_predicates%': len(uniquePredicates(dataset)) / (size(dataset)*2),
        'unique_predicates_T': dataset['tpred'].nunique(),
        'unique_predicates_H': dataset['hpred'].nunique(), 
		'unique_attributes': len(unique_attributes(dataset)),
        'unique_attributes%': len(unique_attributes(dataset)) / (size(dataset)*4),
		'attribute_predicate_rate': len(unique_attributes(dataset)) / len(uniquePredicates(dataset))
	}))

def descriptives(dataset):
    templates = dataset.text.append(dataset.hypothesis).rename('templates')
    predicates = dataset.tpred.append(dataset.hypothesis).rename('predicates')
    return pd.DataFrame([
        templates.value_counts().describe(),
        dataset.text.value_counts().describe(),
        dataset.hypothesis.value_counts().describe(),
        predicates.value_counts().describe(),
        dataset.tpred.value_counts().describe(),
        dataset.hpred.value_counts().describe()
        ])



#TODO: relevance of templates unique to dataset per dataset

#TODO: overlapping templates, attributes, predicates between datasets
#TODO: relevance of templates, attributes, predicates shared by datasets (counts and stuff)

if __name__ == '__main__':
    # TODO: Create utils.io for import/export
    # TODO: Move to separate file
    import os
    INPUT_PATH = os.path.join('..', 'resources', 'datasets')
    OUTPUT_PATH = os.path.join('..', 'resources', 'output')
    headers = ['text', 'hypothesis', 'entailment']
    
    daganlevy = pd.read_csv(os.path.join(INPUT_PATH, 'daganlevy-tidy.csv'))
    zeichner = pd.read_csv(os.path.join(INPUT_PATH, 'zeichner-tidy.csv'))
    dfs = (daganlevy, zeichner)
    datasets = {'daganlevy': daganlevy, 'zeichner': zeichner}
    
    ###
    # Export
    ###

    # Comparison Stats
    df = pd.DataFrame(
        [dataset_stats(dataset) for dataset in dfs], 
        index=datasets)
    df['template_coverage'] = [template_coverage(daganlevy, zeichner), template_coverage(zeichner, daganlevy)]
    df['pred_coverage'] = [pred_coverage(daganlevy, zeichner), pred_coverage(zeichner, daganlevy)]
    df['jaccard_preds'] = [jaccard_index(uniquePredicates(daganlevy), uniquePredicates(zeichner))] * 2
    df['jaccard_templates'] = [jaccard_index(unique_templates(daganlevy), unique_templates(zeichner))] * 2
    df.T.to_csv(os.path.join(OUTPUT_PATH, 'dataset-stats.csv'))
    
    # Descriptives
    for name, dataset in datasets.items():
        outpath = os.path.join(OUTPUT_PATH, name)
        descriptives(dataset).to_csv(outpath + '_descriptives.csv')

