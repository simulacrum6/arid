from baseline import Baseline
import pandas as pd
import os.path as path

# TODO: evaluate via sklearn.metrics
def evaluate(gold, prediction):
    result = ''
    if gold == prediction:
        result += 'T'
    else:
        result += 'F'
    if prediction:
        result += 'P'
    else:
        result += 'N'
    return result

# TODO: abstract to MetricRunner class.
def run_lemma_baseline (dataframes, names=[], output_path=path.join('.')):
    result_headers = ['actual', 'prediction', 'evaluation']
    results = []
    metrics = []
    lemma = Baseline()
    
    for df, name in zip(dataframes, names):
        prediction = lemma.run(df.values)
        evaluation = evaluation = (evaluate(gold, prediction) for gold, prediction in zip(df.entailment, prediction))
        
        #TODO: separate result and confusion
        result = pd.DataFrame(
            list(zip(df.entailment, prediction, evaluation)),
            columns=result_headers)
        result.to_csv(path.join(output_path, ('lemma_baseline-' + name + '.csv')))
        results.append(result)
        
        confusion = result.evaluation.value_counts()
        metric = pd.Series(
            [
                sum(confusion),
                (confusion['TP'] + confusion['TN']) / sum(confusion),
                confusion['TP'] / (confusion['TP'] + confusion['FN']),
                confusion['TP'] / (confusion['TP'] + confusion['FP'])
            ],
            index=['total', 'accuracy', 'recall', 'precision'])
        metric = confusion.append(metric)
        metrics.append(metric)
            
    metrics_df = pd.DataFrame(metrics, index=names)
    metrics_df['f1'] = 2 * metrics_df.recall * metrics_df.precision / (metrics_df.recall + metrics_df.precision)
    metrics_df.to_csv(path.join(output_path, 'lemma_baseline-metrics.csv'))
    
    return [results, metrics]

if __name__ == '__main__':
    input_path = path.join('..', 'resources', 'datasets')
    headers = ['text', 'hypothesis', 'entailment']
    daganlevy = pd.read_json(path.join(input_path, 'daganlevy.json')).reindex(columns=headers).reset_index(drop=True)
    zeichner = pd.read_json(path.join(input_path, 'zeichner.json')).reindex(columns=headers).reset_index(drop=True)
    
    run_lemma_baseline(
        dataframes=[daganlevy, zeichner], 
        names=['daganlevy', 'zeichner'],
        output_path=path.join('..', 'resources', 'output'))