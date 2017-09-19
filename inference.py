import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from utils.qa_utils import get_lemmas_only_verbs, get_lemmas_no_stopwords, get_lemmas, get_lemmas_vo
from utils.representations.embedding import Embedding
import sqlite3
import pandas as pd
import networkx as nx
import sklearn.metrics as skm
from datetime import datetime


LEMMATIZER = WordNetLemmatizer()
STOPWORDS = stopwords.words('english')

class Classifier:
    def run(self, dataset):
        raise NotImplementedError('Classifier.run method not implemented.')


class Baseline(Classifier):
    def __init__(self):
        self.negations = set(['no', 'not', 'never'])
    
    def run(self, test):
        lemma_intersection = np.array([self.lemma_intersection(q, a) for a, q in test])
        matching_voice = np.array([self.matching_voice(q, a) for a, q in test])
        same_negation = np.array([self.same_negation(q, a) for a, q in test])
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


class TypedEntailmentGraph(Classifier):
    def __init__(self, edgelist, typemap):
        self.typemap = typemap
        self.edgelist = [[' '.join(text), ' '.join(hypothesis)] for text,hypothesis in edgelist]
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edgelist)
    
    def run(self, dataset):
        data = self.type_attributes(dataset)
        data = self.lemmatize_predicate(dataset)
        data = self.merge_templates(data)
        print(data)
        return np.array([self.evaluate(text, hypothesis) for text,hypothesis in data])
    
    def type_attributes(self, dataset):
        result = []
        for entry in dataset:
            t,h = entry[0], entry[1] 
            t_typed = [self.type(t[0]), t[1], self.type(t[2])]
            h_typed = [self.type(h[0]), h[1], self.type(h[2])]
            result.append([t_typed, h_typed])
        return result    

    def type(self, string):
        return self.typemap.get(string, string)

    def lemmatize_predicate(self, dataset):
        return [[[t[0], ' '.join(get_lemmas_vo(t[1])), t[2]], [h[0], ' '.join(get_lemmas_vo(h[1])), h[2]]] for t,h in dataset]
    
    def merge_templates(self, dataset):
        return [[' '.join(text), ' '.join(hypothesis)] for text, hypothesis in dataset]
    
    def evaluate(self, text, hypothesis):
        if text in self.graph and hypothesis in self.graph:
            return nx.has_path(self.graph, text, hypothesis)
        else:
            return False  
    


class EntailmentGraph(Classifier):
    def __init__(self, edgelist):
        self.edgelist = edgelist
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edgelist)
    
    def run(self, dataset):
        dataset_lemmas = [[' '.join(get_lemmas_vo(t[1])), ' '.join(get_lemmas_vo(h[1]))] for t,h in dataset]
        return np.array([self.evaluate(text, hypothesis) for text,hypothesis in dataset_lemmas])
    
    def evaluate(self, text, hypothesis):
        if (text in self.graph) and (hypothesis in self.graph):
            return nx.has_path(self.graph, text, hypothesis)
        else:
            return False

class ContextEntailmentGraph(Classifier):
    def __init__(self, edgelist):
        self.edgelist = edgelist
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.edgelist)
    
    def run(self, dataset):
        ds = [[[t[0],' '.join(get_lemmas_vo(t[1])), t[2]], [t[0],' '.join(get_lemmas_vo(h[1])),t[2]]] for t,h in dataset]
        ds = [self.map_arguments(t,h) for t,h in ds]
        return np.array([self.evaluate(t,h) for t,h in ds])
    
    def evaluate(self, text, hypothesis):
        if (text in self.graph) and (hypothesis in self.graph):
            return nx.has_path(self.graph, text, hypothesis)
        else:
            return False
    
    def map_arguments(self, text, hypothesis):
        if text[2] == hypothesis[2]:
            t = ' '.join(['x', text[1], 'y'])
            h = ' '.join(['x', hypothesis[1], 'y'])
            return [t,h]
        else: 
            t = ' '.join(['x', text[1], 'y'])
            h = ' '.join(['y', hypothesis[1], 'x'])
            return [t,h]

        

class Sqlite(Classifier):
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
    

class EmbeddingClassifier(Classifier):
    def __init__(self, embeddingpath):
        self.embedding = Embedding(embeddingpath)
    
    def run(self, dataset):
        return np.array([self.evaluate(t[1],h[1]) for t,h in dataset])
    
    def evaluate(self, word, anotherWord):
        return self.embedding.similarity(word, anotherWord)

class Inclusion(Classifier):  
    def evaluate(self, text, hypothesis):
        t_pred = text[1].split()
        h_pred = hypothesis[1].split()
        return all(word in t_pred for word in h_pred)
    
    def run(self, dataset):
        return np.array([self.evaluate(text, hypothesis) for text, hypothesis in dataset])
    

class RuleMatcher(Classifier):
    def __init__(self, rules, isContextSensitive = False, fuzzy = False):
        self.rules = rules
        self.isContextSensitive = isContextSensitive
        self.fuzzy = fuzzy

    def run(self, dataset):
        if self.isContextSensitive:
            return [self.evaluate(t,h) for t,h in dataset]
        else:
            return [self.evaluate(t[1],h[1]) for t,h in dataset]

    def evaluate(self, text, hypothesis):
        if self.fuzzy:
            return self.fuzzy_match(text, hypothesis)
        else:
            return self.match(text, hypothesis)
            
    def match(self, text, hypothesis):
        for t_rule,h_rule in self.rules:
            if (text == t_rule) and (hypothesis == h_rule):
                return True
            else:
                return False
    
    def fuzzy_match(self, text, hypothesis):
        for t_rule, h_rule in self.rules:
            text_match = self.contains(t_rule, text) or self.contains(text, t_rule)
            hypothesis_match = self.contains(h_rule, hypothesis) or self.contains(h_rule, hypothesis)
            if text_match and hypothesis_match:
                return True
        return False
    
    def contains(self, string, anotherString):
        if string.find(anotherString) == -1:
            return False
        else:
            return True


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

def test_engine():
    from random import random
    import matplotlib.pyplot as plt
    predictions = [
        [True, False, False, False, True, False, False, True, False, False, False, True, True, True, False, True, False, False, False, True],
        [0.5, 0.3, 0.7, 0.38, 0.18, 0.8, 0.3, 0.1, 0.1, 0.6, 0.1, 0.4, 0.7, 0.3, 0.1, 0.01, 0.75, 0.21, 0.5, 0.67]]
    gold = [True, True, True, False, False, False, False, True, True, True, True, False, False, True, False, False, False, True, True, True]
    
    prec, rec, _ = Evaluator.precision_recall_curve(gold, predictions)
    auc = Evaluator.auc(gold, predictions)
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
    import matplotlib.pyplot as plt
    
    outpath = res.output

    datasets = {
        'daganlevy': res.load_dataset('daganlevy', 'analysis'), 
        #'daganlevy_lemmatised': res.load_dataset('daganlevy_lemmatised', 'analysis'),
        'zeichner': res.load_dataset('zeichner', 'analysis')
    }
    gold_annotation = {
        'daganlevy': res.load_dataset('daganlevy', 'tidy').entailment.values,
        'daganlevy_lemmatised': res.load_dataset('daganlevy', 'tidy').entailment.values,
        'zeichner': res.load_dataset('zeichner', 'tidy').entailment.values,
    }
    classifiers = {
        'Lemma Baseline': Baseline(), 
        'Token Subset': Inclusion(), 
        'Entailment Graph': EntailmentGraph(res.load_resource('EntailmentGraph', 'lambda=0.1')), 
        'Relation Embeddings': EmbeddingClassifier('embeddings/relations/words'),
        #'Word Embeddings': EmbeddingClassifier('embeddings/words/words'),
        'PPDB': RuleMatcher(res.load_resource('PPDB2', 'rules'), fuzzy = True),
    }
    
    for name, dataset in datasets.items():
        print('Start classification of {0} @{1}'.format(name, str(datetime.now())))
        result = [classifier.run(dataset) for _,classifier in classifiers.items()]
        result.append(gold_annotation[name])
        
        pd.DataFrame(
            np.transpose(result),
            columns = list(classifiers.keys()) + ['Gold']
        ).to_csv(os.path.join(outpath, name + '_result.csv'))
        print('Done! @{0}'.format(str(datetime.now())))

if __name__ == '__main__':
    test_classifiers()
