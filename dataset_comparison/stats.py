import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

# TODO: lexical difference in preds
# TODO: lemmatised daganlevy

def size(dataset):
	return len(dataset)

def positives(dataset):
	return dataset[dataset['entailment'] == True]

def negatives(dataset):
	return dataset[dataset['entailment'] == False]

def templates(dataset):
    return dataset.text.append(dataset.hypothesis).rename('templates')

def predicates(dataset):
    return dataset.tpred.append(dataset.hpred).rename('predicates')

def attributes(dataset):
    return dataset.tx.append([
        dataset.ty,
        dataset.hx,
        dataset.hy,
    ]).rename('attributes')

def unique_templates(dataset):
    return set(templates(dataset))

def unique_predicates(dataset):
	return set(predicates(dataset))

def unique_attributes(dataset):
	return set(attributes(dataset))

def shared_templates(datasetA, datasetB):
    templatesA = unique_templates(datasetA)
    templatesB = unique_templates(datasetB)
    return templatesA.intersection(templatesB)

def shared_predicates(datasetA, datasetB):
    predicatesA = unique_predicates(datasetA)
    predicatesB = unique_predicates(datasetB)
    return predicatesA.intersection(predicatesB)

def shared_predicates_only(datasetA, datasetB):
    templatesB = unique_predicates(datasetB)
    shared_text = [text in templatesB for text in datasetA.tpred]
    shared_hypothesis = [hypothesis in templatesB for hypothesis in datasetA.hpred]
    shared_temps = [max(t,h) for t,h in zip(shared_text,shared_hypothesis)]
    return datasetA[shared_temps]

def shared_predicates_sorted(datasetA, datasetB):
    predsA = predicates(datasetA).value_counts().to_frame()
    predsB = predicates(datasetB).value_counts().to_frame()
    predsA['rank'] = range(1, predsA.size + 1)
    predsB['rank'] = range(1, predsB.size + 1)
    C = predsA.join(predsB, lsuffix = 'A', rsuffix = 'B')
    C.dropna(inplace = True)
    C['avgrank'] = (C['rankA'] + C['rankB']) /  2
    C['freq'] = C['predicatesA'] + C['predicatesB']
    C.sort_values('avgrank', inplace = True)
    return C[['avgrank', 'freq']]

def shared_attributes(datasetA, datasetB):
    attributesA = unique_attributes(datasetA)
    attributesB = unique_attributes(datasetB)
    return attributesA.intersection(attributesB)

#TODO: create abstracted coverage(A,B) method
def coverage_templates(datasetA, datasetB):
    templatesA = unique_templates(datasetA)
    templatesB = unique_templates(datasetB)    
    return len(templatesA & templatesB) / len(templatesB)

def coverage_predicates(datasetA, datasetB):
	predsA = unique_predicates(datasetA)
	predsB = unique_predicates(datasetB)
	return len(predsA & predsB) / len(predsB)

def coverage_attributes(datasetA, datasetB):
	attsA = unique_attributes(datasetA)
	attsB = unique_attributes(datasetB)
	return len(attsA & attsB) / len(attsB)

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
        'unique_predicates': len(unique_predicates(dataset)),
        'unique_predicates%': len(unique_predicates(dataset)) / (size(dataset)*2),
        'unique_predicates_T': dataset['tpred'].nunique(),
        'unique_predicates_H': dataset['hpred'].nunique(), 
		'unique_attributes': len(unique_attributes(dataset)),
        'unique_attributes%': len(unique_attributes(dataset)) / (size(dataset)*4),
		'attribute_predicate_rate': len(unique_attributes(dataset)) / len(unique_predicates(dataset))
	}))


def comparison_stats(datasetA, datasetB):
    return pd.Series(OrderedDict({
        'shared_templates': len(shared_templates(datasetA, datasetB)),
        'shared_predicates': len(shared_predicates(datasetA, datasetB)),
        'shared_attributes': len(shared_attributes(datasetA, datasetB)),
        'coverage_templates': coverage_templates(datasetA, datasetB),
        'coverage_predicates': coverage_predicates(datasetA, datasetB),
        'coverage_attributes': coverage_attributes(datasetA, datasetB),
        'jaccard_templates': jaccard_index(unique_templates(datasetA), unique_templates(datasetB)),
        'jaccard_predicates': jaccard_index(unique_predicates(datasetA), unique_predicates(datasetB)),
        'jaccard_attributes': jaccard_index(unique_attributes(datasetA), unique_attributes(datasetB))
    }))


def descriptives(dataset):
    temps = templates(dataset)
    preds = predicates(dataset)
    attrs = attributes(dataset)
    return pd.DataFrame([
        temps.value_counts().describe(),
        dataset.text.value_counts().describe(),
        dataset.hypothesis.value_counts().describe(),
        preds.value_counts().describe(),
        dataset.tpred.value_counts().describe(),
        dataset.hpred.value_counts().describe(),
        attrs.value_counts().describe(),
        dataset.tx.append(dataset.ty).rename('attributes_text').value_counts().describe(),
        dataset.hx.append(dataset.hy).rename('attributes_hypothesis').value_counts().describe()
        ])
    

def top10(series):
    top = series.value_counts().head(10)
    df = pd.DataFrame(
        list(zip(top.index, top.values)),
        columns = [top.name, 'frequency'])
    df.reset_index(drop = True, inplace = True)
    return df

def top10s(dataset):
    return pd.concat([
        top10(templates(dataset)),
        top10(predicates(dataset)),
        top10(attributes(dataset))
        ], 
        axis = 1)

#TODO: Optional relevance of templates unique to dataset per dataset
#TODO: Optional relevance of templates, attributes, predicates shared by datasets (counts and stuff)

#TODO: duplicates in hypothesis parts, daganlevy
# poland was divided among russia 4690,4959
# jupiter is big as the earth 1989, 13983, 14006

if __name__ == '__main__':
    # TODO: Create utils.io for import/export
    # TODO: Move to separate file
    import os
    INPUT_PATH = os.path.join('..', 'resources', 'datasets')
    OUTPUT_PATH = os.path.join('..', 'resources', 'output')
    
    # Prepare inputs
    daganlevy = pd.read_csv(os.path.join(INPUT_PATH, 'daganlevy-tidy.csv'))
    zeichner = pd.read_csv(os.path.join(INPUT_PATH, 'zeichner-tidy.csv'))
    dfs = (daganlevy, zeichner)
    datasets = {'daganlevy': daganlevy, 'zeichner': zeichner}
    
    ###
    # Comparison and export
    ###
    
    # Comparison Stats
    stats_ds = pd.DataFrame(
        [dataset_stats(dataset) for dataset in dfs], 
        index=['daganlevy', 'zeichner'])
    stats_cmp = pd.DataFrame(
        [comparison_stats(daganlevy, zeichner), comparison_stats(zeichner, daganlevy)],
        index = ['daganlevy', 'zeichner'])
    pd.concat([stats_ds, stats_cmp], axis = 1).T.to_csv(os.path.join(OUTPUT_PATH, 'dataset-stats.csv'))
    
    # top10 shared among datasets
    daganlevy_shared = shared_predicates_only(daganlevy, zeichner)
    zeichner_shared = shared_predicates_only(zeichner, daganlevy)
    all_shared_preds = shared_predicates_sorted(daganlevy, zeichner)
    
    top10s(daganlevy_shared).to_csv(os.path.join(OUTPUT_PATH, 'daganlevy_shared_top10.csv'))
    top10s(zeichner_shared).to_csv(os.path.join(OUTPUT_PATH, 'zeichner_shared_top10.csv'))
    all_shared_preds.head(10).to_csv(os.path.join(OUTPUT_PATH, 'top10_shared_predicates.csv'))
    
    for name, dataset in datasets.items():
        outpath = os.path.join(OUTPUT_PATH, name)
        descriptives(dataset).to_csv(outpath + '_descriptives.csv')
        top10s(dataset).to_csv(outpath + '_top10.csv')
