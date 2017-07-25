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

def predicates(dataset):
    return dataset.tpred.append(dataset.hpred).rename('predicates')

def templates(dataset):
    return dataset.text.append(dataset.hypothesis).rename('templates')

def attributes(dataset):
    return dataset.tx.append([
        dataset.ty,
        dataset.hx,
        dataset.hy,
    ]).rename('attributes')

def unique_predicates(dataset):
	return set(predicates(dataset))

# TODO: lexical difference in preds

# TODO: lemmatised daganlevy

def unique_attributes(dataset):
	return set(dataset['ty'].append([
		dataset['tx'], 
		dataset['hx'], 
		dataset['hy']]))

def unique_templates(dataset):
    return set(templates(dataset))

#TODO: create abstracted coverage(A,B) method
def pred_coverage(datasetA, datasetB):
	predsA = unique_predicates(datasetA)
	predsB = unique_predicates(datasetB)
	return len(predsA & predsB) / len(predsB)

def template_coverage(datasetA, datasetB):
    templatesA = unique_templates(datasetA)
    templatesB = unique_templates(datasetB)    
    return len(templatesA & templatesB) / len(templatesB)

def shared(datasetA, datasetB, column):
    A = set(datasetA[column])
    B = set(datasetB[column])
    return A.intersection(B)

def shared_templates(datasetA, datasetB):
    templatesA = unique_templates(datasetA)
    templatesB = unique_templates(datasetB)
    return templatesA.intersection(templatesB)

def shared_predicates_in(datasetA, datasetB):
    templatesB = unique_predicates(datasetB)
    shared_text = [text in templatesB for text in datasetA.tpred]
    shared_hypothesis = [hypothesis in templatesB for hypothesis in datasetA.hpred]
    shared_temps = [max(t,h) for t,h in zip(shared_text,shared_hypothesis)]
    return datasetA[shared_temps]

def shared_predicates_sorted(datasetA, datasetB):
    predsA = predicates(datasetA).value_counts().to_frame()
    predsA['rank'] = range(1, predsA.size + 1)
    predsB = predicates(datasetB).value_counts().to_frame()
    predsB['rank'] = range(1, predsB.size + 1)
    C = predsA.join(predsB, lsuffix = 'A', rsuffix = 'B')
    C.dropna(inplace = True)
    C['avgrank'] = (C['rankA'] + C['rankB']) /  2
    C['freq'] = C['predicatesA'] + C['predicatesB']
    C.sort_values('avgrank', inplace = True)
    return C[['avgrank', 'freq']]


def shared_predicates(datasetA, datasetB):
    predicatesA = unique_predicates(datasetA)
    predicatesB = unique_predicates(datasetB)
    return predicatesA.intersection(predicatesB)

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
    return pd.DataFrame(series.value_counts()[:10])

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
    df = pd.DataFrame(
        [dataset_stats(dataset) for dataset in dfs], 
        index=['daganlevy', 'zeichner'])
    df['shared_preds'] = [len(shared_predicates(daganlevy, zeichner))] * 2
    df['shared_templates'] = [len(shared_templates(daganlevy, zeichner))] * 2
    df['template_coverage'] = [template_coverage(daganlevy, zeichner), template_coverage(zeichner, daganlevy)]
    df['pred_coverage'] = [pred_coverage(daganlevy, zeichner), pred_coverage(zeichner, daganlevy)]
    df['jaccard_preds'] = [jaccard_index(unique_predicates(daganlevy), unique_predicates(zeichner))] * 2
    df['jaccard_templates'] = [jaccard_index(unique_templates(daganlevy), unique_templates(zeichner))] * 2
    df.T.to_csv(os.path.join(OUTPUT_PATH, 'dataset-stats.csv'))
    
    # top10 shared among datasets
    daganlevy_shared = shared_predicates_in(daganlevy, zeichner)
    top10(templates(daganlevy_shared)).to_csv(os.path.join(OUTPUT_PATH, 'daganlevy_top10_shared_predicates(templateview).csv'))
    top10(predicates(daganlevy_shared)).to_csv(os.path.join(OUTPUT_PATH, 'daganlevy_top10_shared_predicates.csv'))
    top10(attributes(daganlevy_shared)).to_csv(os.path.join(OUTPUT_PATH, 'daganlevy_top10_shared_attributes.csv'))
    
    zeichner_shared = shared_predicates_in(zeichner, daganlevy)
    top10(templates(zeichner_shared)).to_csv(os.path.join(OUTPUT_PATH, 'zeichner_top10_shared_predicates(templateview).csv'))
    top10(predicates(zeichner_shared)).to_csv(os.path.join(OUTPUT_PATH, 'zeichner_top10_shared_predicates.csv'))
    top10(attributes(zeichner_shared)).to_csv(os.path.join(OUTPUT_PATH, 'zeichner_top10_shared_attributes.csv'))
    
    all_shared = shared_predicates_sorted(daganlevy, zeichner)
    all_shared.head(10).to_csv(os.path.join(OUTPUT_PATH, 'top10_shared_predicates.csv'))
    
    for name, dataset in datasets.items():
        outpath = os.path.join(OUTPUT_PATH, name)
        # Descriptives
        descriptives(dataset).to_csv(outpath + '_descriptives.csv')
        # Top10s
        outpath = os.path.join(OUTPUT_PATH, 'top10s', name)
        top10(templates(dataset)).to_csv(outpath + '_top10_templates.csv')
        top10(dataset.text).to_csv(outpath + '_top10_texts.csv')
        top10(dataset.hypothesis).to_csv(outpath + '_top10_hypothesis.csv')
        top10(predicates(dataset)).to_csv(outpath + '_top10_predicates.csv')
        top10(dataset.tpred).to_csv(outpath + '_top10_tpreds.csv')
        top10(dataset.hpred).to_csv(outpath + '_top10_hpreds.csv')
        top10(attributes(dataset)).to_csv(outpath + '_top10_attributes.csv')
        top10(dataset.tx.append(dataset.ty).rename('attributes_text')).to_csv(outpath + '_top10_attributes_t.csv')
        top10(dataset.hx.append(dataset.hy).rename('attributes_hypothesis')).to_csv(outpath + '_top10_attributes_h.csv')
    
