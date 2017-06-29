import pandas as pd

def size(dataframe):
	return len(dataframe)

def positives(dataframe):
	return (dataframe['entailment'] == True).size

def negatives(dataframe):
	return (dataframe['entailment'] == False).size

# TODO: split into hypothesis and text
def uniquePredicates(dataframe):
	return dataframe.text_predicate.append(dataframe.hypothesis_predicate).value_counts().size

def uniqueAttributes(dataframe):
	return dataframe.text_y.append([
		dataframe.text_x, 
		dataframe.hypothesis_x, 
		dataframe.hypothesis_y]).value_counts().size

def dataframe_stats(dataframe):
	return pd.Series({
		'size': size(dataframe),
		'positives': positives(dataframe),
		'negatives': negatives(dataframe),
		'uniquePredicates': uniquePredicates(dataframe),
		'uniqueAttributes': uniqueAttributes(dataframe),
		'attributes_per_predicate': uniqueAttributes(dataframe) / uniquePredicates(dataframe)
	})

def ruleCoverage(df1, df2):
	#TODO: Implement
	return None


# TODO: Create utils.io for import/export
# TODO: Move to separate file
INPUT_PATH = '..\\resources\\datasets\\'
OUTPUT_PATH = '..\\resources\\output\\'
headers = ['text', 'hypothesis', 'entailment']

daganlevy = pd.read_csv(INPUT_PATH + 'daganlevy-tidy.csv')
zeichner = pd.read_csv(INPUT_PATH + 'zeichner-tidy.csv')
dfs = (daganlevy, zeichner)

df = pd.DataFrame(
	[dataframe_stats(dataframe) for dataframe in dfs], 
	index=['daganlevy', 'zeichner'])
df.to_csv(OUTPUT_PATH + 'dataframe-stats.csv')