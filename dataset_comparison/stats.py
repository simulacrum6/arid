import pandas as pd

def size(dataset):
	return dataset.size

def positives(dataset):
	return dataset[dataset['entailment'] == True]

def negatives(dataset):
	return dataset[dataset['entailment'] == False]

# TODO: split into hypothesis and text
def uniquePredicates(dataset):
	return set(dataset['tpred'].append(dataset['hpred']))

def uniqueAttributes(dataset):
	return set(dataset['ty'].append([
		dataset['tx'], 
		dataset['hx'], 
		dataset['hy']]))

def uniqueRules(dataset):
    return set(dataset['text'].append(dataset['hypothesis']))

#TODO: create abstracted coverage(A,B) method
def predCoverage(datasetA, datasetB):
	predsA = uniquePredicates(datasetA)
	predsB = uniquePredicates(datasetB)
	return len(predsA & predsB) / len(predsB)

def ruleCoverage(datasetA, datasetB):
    rulesA = uniqueRules(datasetA)
    rulesB = uniqueRules(datasetB)    
    return len(rulesA & rulesB) / len(rulesB)

def jaccardIndex(list1, list2):
    #FIXME: implement
    return -1


#TODO: return ordered dict
def dataset_stats(dataset):
	return pd.Series({
		'size': size(dataset),
		'positives': len(positives(dataset)),
		'negatives': len(negatives(dataset)),
		'pn_rate': len(positives(dataset)) / len(negatives(dataset)),
        'uniqueRules': len(uniqueRules(dataset)),
		'uniquePredicates': len(uniquePredicates(dataset)),
        'uniquePredicates%': len(uniquePredicates(dataset)) / size(dataset), 
		'uniqueAttributes': len(uniqueAttributes(dataset)),
        'uniqueAttributes%': len(uniqueAttributes(dataset)) / size(dataset),
		'attributes_per_predicate': len(uniqueAttributes(dataset)) / len(uniquePredicates(dataset))
	})


# TODO: Create utils.io for import/export
# TODO: Move to separate file
INPUT_PATH = '..\\resources\\datasets\\'
OUTPUT_PATH = '..\\resources\\output\\'
headers = ['text', 'hypothesis', 'entailment']

daganlevy = pd.read_csv(INPUT_PATH + 'daganlevy-tidy.csv')
zeichner = pd.read_csv(INPUT_PATH + 'zeichner-tidy.csv')
dfs = (daganlevy, zeichner)

# export
df = pd.DataFrame(
	[dataset_stats(data) for data in dfs], 
	index=['daganlevy', 'zeichner'])
df.to_csv(OUTPUT_PATH + 'dataset-stats.csv')