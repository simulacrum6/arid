from baseline import Baseline
import pandas as pd

# TODO: scale for multiple datasets

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

headers = ['text', 'hypothesis', 'entailment']
result_headers = ['actual', 'prediction', 'evaluation']
lemma = Baseline()

### Dagan & Levy run

daganlevy = pd.read_json('..\\resources\\datasets\\daganlevy.json').reindex(columns=headers).reset_index(drop=True)

prediction = lemma.run(daganlevy.values)
evaluation = (evaluate(gold, prediction) for gold, prediction in zip(daganlevy.entailment, prediction))

result = pd.DataFrame(
    list(zip(daganlevy.entailment, prediction, evaluation)),
    columns=result_headers)

confusion = result.evaluation.value_counts()

acc = (confusion['TP'] + confusion['TN']) / sum(confusion)
recall = confusion['TP'] / (confusion['TP'] + confusion['FN'])
precision = confusion['TP'] / (confusion['TP'] + confusion['FP'])


### Zeichner run

zeichner = pd.read_json('..\\resources\\datasets\\zeichner.json').reindex(columns=headers).reset_index(drop=True)

prediction = lemma.run(zeichner.values)
evaluation = (evaluate(gold, prediction) for gold, prediction in zip(zeichner.entailment, prediction))

result = pd.DataFrame(
    list(zip(zeichner.entailment, prediction, evaluation)),
    columns=result_headers)

confusion = result.evaluation.value_counts()

acc = (confusion['TP'] + confusion['TN']) / sum(confusion)
recall = confusion['TP'] / (confusion['TP'] + confusion['FN'])
precision = confusion['TP'] / (confusion['TP'] + confusion['FP'])