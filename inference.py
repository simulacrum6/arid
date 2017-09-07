import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from utils.qa_utils import get_lemmas_only_verbs, get_lemmas_no_stopwords, get_lemmas
import sqlite3
import pandas as pd
import networkx as nx
import sklearn.metrics as skm

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = stopwords.words('english')

class Classifier:
    def run(self, dataset):
        raise NotImplementedError('Classifier.run method not implemented.')


class Baseline(Classifier):
    def __init__(self):
        self.negations = set(['no', 'not', 'never'])
    
    def run(self, test):
        lemma_intersection = np.array([self.lemma_intersection(q, a) for q, a in test])
        matching_voice = np.array([self.matching_voice(q, a) for q, a in test])
        same_negation = np.array([self.same_negation(q, a) for q, a in test])
        return lemma_intersection * matching_voice * same_negation
    
    @staticmethod
    def lemma_intersection(q, a):
        q_lemmas_only_verbs = get_lemmas_only_verbs(q[1])
        a_lemmas_only_verbs = get_lemmas_only_verbs(a[1])
        q_lemmas_no_stopwords = get_lemmas_no_stopwords(q[1])
        a_lemmas_no_stopwords = get_lemmas_no_stopwords(a[1])
    
        share_one_verb = len(q_lemmas_only_verbs.intersection(a_lemmas_only_verbs)) > 0
        answer_contains_all_contents = q_lemmas_no_stopwords == q_lemmas_no_stopwords.intersection(a_lemmas_no_stopwords)
        return share_one_verb and answer_contains_all_contents
    
    def matching_voice(self, q, a):
        return self.same_voice(q, a) == self.aligned_args(q, a)
    
    def same_voice(self, q, a):
        q_passive = self.is_passive(q[1])
        a_passive = self.is_passive(a[1])
        return q_passive == a_passive
    
    @staticmethod
    def is_passive(pred):
        words = get_lemmas(pred)
        be = 'be' in words
        by = 'by' in words
        return be and by
    
    @staticmethod
    def aligned_args(q, a):
        q_arg = get_lemmas_no_stopwords(q[2], wn.NOUN)
        if q_arg == get_lemmas_no_stopwords(a[2], wn.NOUN):
            return True
        if q_arg == get_lemmas_no_stopwords(a[0], wn.NOUN):
            return False
        raise Exception('HORRIBLE BUG!!!')
    
    def same_negation(self, q, a):
        q_negated = self.is_negated(q[1])
        a_negated = self.is_negated(a[1])
        return q_negated == a_negated
    
    def is_negated(self, pred):
        words = get_lemmas(pred)
        return len(set(words).intersection(self.negations)) > 0


# TODO: allow non exact matching mode by matching substrings in x, pred, y
class EntailmentGraph(Classifier):
    def __init__(self, edgelist, typemap):
        self.typemap = typemap
        self.edgelist = [[' '.join(text), ' '.join(hypothesis)] for text,hypothesis in edgelist]
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edgelist)
    
    def run(self, dataset):
        data = self.type_attributes(dataset)
        data = self.merge_templates(data)
        return np.array([self.evaluate(text, hypothesis) for text,hypothesis in data])
    
    def type_attributes(self, dataset):
        result = []
        for rule in dataset:
             result.append([[self.type(x), pred, self.type(y)] for x,pred,y in rule])
    
        return result    
            
    def merge_templates(self, dataset):
        return [[' '.join(text), ' '.join(hypothesis)] for text, hypothesis in dataset]
    
    def type(self, string):
        return self.typemap.get(string, string)
    
    def evaluate(self, text, hypothesis):
        if text in self.graph and hypothesis in self.graph:
            return nx.has_path(self.graph, text, hypothesis)
        else:
            return False  
    

class EntailmentGraph2(Classifier):
    def __init__(self, edgelist, typemap):
        self.typemap = typemap
        self.edgelist = [[text[1], hypothesis[1]] for text, hypothesis in edgelist]
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edgelist)
    
    def run(self, dataset):
        data = self.type_attributes(dataset)
        data = self.merge_templates(data)
        return np.array([self.evaluate(text, hypothesis) for text,hypothesis in data])
    
    def type_attributes(self, dataset):
        result = []
        for rule in dataset:
             result.append([[self.type(x), pred, self.type(y)] for x,pred,y in rule])
        return result
    
    @staticmethod
    def lemmatize(phrase):
        return ' '.join([LEMMATIZER.lemmatize(word) for word in phrase])
    
    def merge_templates(self, dataset):
        return [[self.lemmatize(text[1]), self.lemmatize(hypothesis[1])] for text, hypothesis in dataset]
    
    def type(self, string):
        nostop = ' '.join([word for word in string.split() if word not in STOPWORDS])
        return self.typemap.get(nostop, nostop)
        
    def evaluate(self, text, hypothesis):
        if (text in self.graph) and (hypothesis in self.graph):
            return nx.has_path(self.graph, text, hypothesis)
        else:
            return False 
            

class PPDB2(Classifier):
    def __init__(self, dbpath):
        self.db = dbpath
        self.con = sqlite3.connect(dbpath)
    
    def run(self, dataset):
        self.con = sqlite3.connect(self.db)
        self.write_to_db(dataset)
        matches = self.find_paraphrases()
        self.clean_db()
        return np.array([self.evaluate(match) for match in matches])
    
    def write_to_db(self, dataset):
        df = pd.DataFrame(dataset, columns=['text', 'hypothesis'])
        df['tx'], df['tpred'], df['ty'] = zip(*df.text)
        df['hx'], df['hpred'], df['hy'] = zip(*df.hypothesis)
        df.drop(['text', 'hypothesis'], axis=1, inplace=True)
        df.to_sql('data', self.con, if_exists='replace')
    
    def clean_db(self):
        self.con.execute('DROP TABLE data')
        self.con.close()
    
    @staticmethod
    def evaluate(match):
        if match:
            return True
        else:
            return False
    
    def find_paraphrases(self):
        return pd.read_sql_query(
            'SELECT paraphrases.entailment ' 
            + 'FROM  data ' 
            + 'LEFT JOIN paraphrases ' 
            + 'ON data.tpred = paraphrases.phrase ' 
            + 'AND data.hpred = paraphrases.paraphrase', 
            self.con).entailment.values
    


class ClassificationEngine:
    """Engine for running a set of relation inference classifiers over a dataset.
    
    Methods:
        run(classifiers, dataset) -- Runs a list of classifiers over dataset
        
        Parameters:
            classifiers -- Listlike of classifiers as arid.inference.classifiers.Classifier
            dataset -- Dataset to be classified as numpy.ndarray (2,3). Should contain Text and Hypothesis and X attribute, predicate, Y attribute for each
    
        Returns:
            A list of numpy.ndarrays (shape=(len(classifiers),len(dataset)) dtype='float'). 
            The list contains one array for each classifier. Each array contains either True/False for each entry in the dataset.
    """
    
    def __init__(self, classifiers):
        self.classifiers = classifiers
    
    def run(self, dataset):
        """Run a list of classifiers over dataset. See help(ClassificationEngine) for details"""
        return [cf.run(dataset) for cf in self.classifiers]


class Inclusion(Classifier):  
    def evaluate(self, text, hypothesis):
        t_pred = text[1].split()
        h_pred = hypothesis[1].split()
        return all(word in t_pred for word in h_pred)
    
    def run(self, dataset):
        return [self.evaluate(text, hypothesis) for text, hypothesis in dataset]
    

class Evaluator:
    def aggregate(self, predictions, function):
        """Aggregates each sublist of predictions, using the specified function"""
        return [function(sublist) for sublist in np.transpose(predictions)]
    
    def precision_recall(self, gold, prediction):
        prediction = self.aggregate(prediction, max)
        return skm.precision_recall_curve(gold, prediction)
    
    def auc(self, gold, prediction):
        precision, recall, _ = self.precision_recall(gold, prediction)
        return skm.auc(recall, precision)
    

def test_engine():
    from random import random
    import matplotlib.pyplot as plt
    predictions = [
        [True, False, False, False, True, False, False, True, False, False, False, True, True, True, False, True, False, False, False, True],
        [0.5, 0.3, 0.7, 0.38, 0.18, 0.8, 0.3, 0.1, 0.1, 0.6, 0.1, 0.4, 0.7, 0.3, 0.1, 0.01, 0.75, 0.21, 0.5, 0.67]]
    gold = [True, True, True, False, False, False, False, True, True, True, True, False, False, True, False, False, False, True, True, True]
    
    evaluator = Evaluator()
    prec, rec, _ = evaluator.precision_recall(gold, predictions)
    auc = evaluator.auc(gold, predictions)
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0,1.05])
    plt.ylim([0,1.05])
    
    plt.show()

def test_classifiers():
    import os
    import utils.resources as res
    import datetime as dt
    
    outpath = res.output
    
    baseline = Baseline()
    graph = EntailmentGraph2(
        res.load_resource('EntailmentGraph', 'edgelist'),
        res.load_resource('EntailmentGraph', 'typemap')
        )
    ppdb = PPDB2(res.load_resource('PPDB2', 'db-mini'))
    inc = Inclusion()
    
    daganlevy = res.load_dataset('daganlevy', 'analysis')
    zeichner = res.load_dataset('zeichner', 'analysis')
    
    datasets = {
        'daganlevy': daganlevy, 
        'zeichner': zeichner
        }
       
    gold_annotation = {
        'daganlevy': res.load_dataset('daganlevy', 'tidy').entailment.values,
        'zeichner': res.load_dataset('zeichner', 'tidy').entailment.values,
    }
    classifiers = [baseline, inc, graph, ppdb]
    evaluator = Evaluator()
    
    for name, dataset in datasets.items():
        print('Start classification of ' + name)
        result = [] 
        
        for classifier in classifiers:
            print('Start Classification Task @' + dt.datetime.now().isoformat())
            result.append(classifier.run(dataset))
            print('Done @' + dt.datetime.now().isoformat())
        
        pd.DataFrame(
            np.transpose(result)
            #columns=['baseline', 'entailment_graph', 'ppdb']
            ).to_csv(os.path.join(outpath, name + '_result.csv'))
        
        auc = evaluator.auc(gold_annotation[name], result)
        print('Auc({0}): {1}'.format(name, auc))


if __name__ == '__main__':
    test_engine()
    test_classifiers()