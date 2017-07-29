import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

INPUT_PATH = os.path.join('..', 'resources', 'datasets')
OUTPUT_PATH = os.path.join('..', 'resources', 'output')

# TODO: overlap lemmatised/not lemmatised -> examples
# TODO: lexical difference in preds
# TODO: duplicates in hypothesis parts, daganlevy
# poland was divided among russia 4690,4959
# jupiter is big as the earth 1989, 13983, 14006

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

def coverage(datasetA, datasetB, coverage_function):
    A = coverage_function(datasetA)
    B = coverage_function(datasetB)
    return len(A & B) / len(B)

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

def compare_datasets(datasetA, datasetB, names = ['A', 'B']):
    datasets = [datasetA, datasetB]
    stats = pd.DataFrame(
            [
                dataset_stats(datasetA).append(comparison_stats(datasetA, datasetB)),
                dataset_stats(datasetB).append(comparison_stats(datasetB, datasetA))
            ], 
            index=names)
    shared_preds = shared_predicates_sorted(datasetA, datasetB)
        
    for i, name in enumerate(names):
        outpath = os.path.join(OUTPUT_PATH, name)
        descriptives(datasets[i]).to_csv(outpath + '_descriptives.csv')
        top10s(datasets[i]).to_csv(outpath + '_top10.csv')
        top10s(shared_predicates_only(datasets[i], datasets[i-1])).to_csv(outpath + '_top10-shared.csv')
    
    suffix = '_'.join(names) + '.csv' 
    stats.T.to_csv(os.path.join(OUTPUT_PATH, 'dataset_stats-' + suffix))
    shared_preds.to_csv(os.path.join(OUTPUT_PATH, 'shared_predicates-' + suffix))

if __name__ == '__main__':
    daganlevy = pd.read_csv(os.path.join(INPUT_PATH, 'daganlevy-tidy.csv'))
    daganlevy_lemmatised = pd.read_csv(os.path.join(INPUT_PATH, 'daganlevy-tidy_lemmatised.csv'))
    zeichner = pd.read_csv(os.path.join(INPUT_PATH, 'zeichner-tidy.csv'))
    
    compare_datasets(daganlevy, zeichner, names = ['daganlevy', 'zeichner'])
    compare_datasets(daganlevy_lemmatised, zeichner, names = ['daganlevy_lemmatised', 'zeichner'])
