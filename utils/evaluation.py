import sklearn.metrics as skm
import resources as res
import pandas as pd
import numpy as np
import random as rand
import os

class Evaluator:
    @staticmethod
    def aggregate(predictions, aggfunc):
        """Aggregates each sublist of predictions, using the specified function"""
        return [aggfunc(sublist) for sublist in np.transpose(predictions)]
    
    @staticmethod
    def precision_recall_curve(gold, predictions):
        prediction = Evaluator.aggregate(predictions, max)
        return skm.precision_recall_curve(gold, prediction)
    
    @staticmethod
    def auc(gold, predictions):
        precision, recall, _ = Evaluator.precision_recall_curve(gold, predictions)
        return skm.auc(recall, precision)

    @staticmethod
    def delta_auc(gold, predictions):
        prediction = Evaluator.aggregate(predictions,max)
        auc_baseline = sum(gold)/len(gold)
        auc_prediction = skm.average_precision_score(gold, prediction)
        return auc_prediction - auc_baseline
    
    @staticmethod
    def avp(gold, predictions):
        return skm.average_precision_score(gold, Evaluator.aggregate(predictions, max))
    
    @staticmethod
    def fp_fn(gold, predictions):
        return [[self.fasle_positives(gold,p), false_negatives(gold,p)] for p in predictions]
    
    @staticmethod
    def false_positives(gold, prediction):
        return [(not bool(g) and bool(p)) for g,p in zip(gold, prediction)]
    
    @staticmethod
    def false_negatives(gold, prediction):
        return [(bool(g) and (not bool(p))) for g,p in zip(gold, prediction)]

    @staticmethod
    def base_stats(gold, prediction):
        scores = [
            skm.recall_score,
            skm.precision_score,
            skm.accuracy_score,
            skm.f1_score,
            skm.average_precision_score
        ]
        return [score(gold, prediction) for score in scores]
        

def get_sample(result, samplesize=3000, positive_rate = 0.5):
    positives = result[result['Gold'] == 1]
    negatives = result[result['Gold'] == 0]
    pos = rand.sample(list(positives.index), int(samplesize*positive_rate))
    neg = rand.sample(list(negatives.index), int(samplesize*(1 - positive_rate)))
    pos.extend(neg)
    return result.iloc[pos]

def get_samples(result, samplesize, positive_rate = 0.5, n=100):
    return [get_sample(result, samplesize, positive_rate) for _ in range(iterations)]

def calc_base_stats(results):
    classifiers = [
        'Lemma Baseline',
        'Entailment Graph',
        'PPDB'
    ]
    stat_names = [
        'Rec.', 
        'Prec.', 
        'Acc.', 
        'F1',
        'AUC'
    ]
    for name, result in results.items():
        gold = results[name]['Gold'].values
        predictions = results[name][classifiers].T.values
        stats = [Evaluator.base_stats(gold, prediction) for prediction in predictions]
        pd.DataFrame(stats, index=classifiers, columns=stat_names).to_csv(path.join(res.output,'stats','base_stats-{0}.csv'.format(name)))

def add_prediction(datasets, results, classifiers):
    for name, dataset in datasets.items():
        result = results[name]
        for method, classifier in classifiers.items():
            prediction = classifier.run(dataset)
            prediction = [float(x) for x in prediction]
            result[method] = prediction
        result.to_csv(os.path.join(
            res.output,
            '{0}_result.csv'.format(name)
        ))
