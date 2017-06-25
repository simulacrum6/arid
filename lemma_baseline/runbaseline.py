import baseline as bl
import pandas as pd
import csv
    
def evaluate(goldstandard, prediction):
    result = ''
    if goldstandard == prediction:
        result += 'T'
    else:
        result += 'F'
    if prediction:
        result += 'P'
    else:
        result += 'N'
    return result

annotationMap = {
    'y': True,
    'n': False
}

data = pd.read_csv(
    'daganlevy_dataset.txt',
    sep='\t',
    header=None,
    names=['text', 'hypothesis', 'annotation'])

for i in ['text', 'hypothesis']:
    data[i] = [template.split(', ') for template in data[i]]

exp = bl.Baseline()
baseline = exp.run(data.values)

data['value'] = [annotationMap[annotation] for annotation in data['annotation']]
data['baseline_prediction'] = baseline

data['evaluation'] = [evaluate(gold, prediction) for gold, prediction in zip(data['value'], data['baseline_prediction'])]

confusion = data['evaluation'].value_counts()

acc = (confusion['TP'] + confusion['TN']) / sum(confusion)
recall = confusion['TP'] / (confusion['TP'] + confusion['FN'])
precision = confusion['TP'] / (confusion['TP'] + confusion['FP'])