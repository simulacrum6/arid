import numpy as np
from nltk.corpus import wordnet as wn
from qa_utils import get_lemmas_only_verbs, get_lemmas_no_stopwords, get_lemmas
import sqlite3
import pandas as pd

class Classifier:
    def run(self, dataset):
        raise NotImplementedError('Classifier.run method not implemented.')

class Baseline(Classifier):
    def __init__(self):
        self.negations = set(['no', 'not', 'never'])
    
    # removed v in q,a,v. might break
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
        self.edgelist = np.transpose(edgelist)
    
    def run(self, dataset):
        self.type_attributes(dataset)
        return np.array([self.evaluate(text, hypothesis) for text,hypothesis in dataset])
    
    def type_attributes(self, dataset):
        dataset[0] = np.array([self.type(x), pred, self.type(y)] for x,pred,y in dataset[0])
        dataset[1] = np.array([self.type(x), pred, self.type(y)] for x,pred,y in dataset[1])
    
    def type(self, string):
        if self.typemap[string]:
            return self.typemap[string]
        else:
            return string
    
    # exact matching
    def evaluate(self, text, hypothesis):
        if (text, hypothesis) in self.edgelist:
            return True
        else:
            return False

class PPDB2(Classifier):
    def __init__(self, dbpath):
        self.con = sqlite3.connect(dbpath)
    
    def run(self, dataset):
        self.write_to_db(dataset)
        self.clean_up()
        return np.array(self.evaluate())
    
    def write_to_db(self, dataset):
        df = pd.DataFrame(dataset, columns=['text', 'hypothesis'])
        df['tx'], df['tpred'], df['ty'] = zip(*df.text)
        df['hx'], df['hpred'], df['hy'] = zip(*df.hypothesis)
        df.to_sql('data', self.con, if_exists='replace')
    
    def clean_up(self):
        self.con.execute('DROP TABLE data')
        self.con.close()
    
    def evaluate(self):     
        result = pd.read_sql_query(
            'SELECT paraphrases.entailment ' +
            'FROM  paraphrases ' +
            'JOIN data ' + 
            'ON data.tpred = paraphrases.phrase ' + 
            'AND data.hpred = paraphrases.paraphrase', con)
        
        def relabel(sqlresult):
            if sqlresult:
                return True
            else:
                return False
        
        return [relabel(entry) for entry in result.entailment]



def main():
    #temporary solution
    import sys, os
    sys.path.append('C:\\Users\\Nev\\Projects')
    from arid.utils import resources as res
    
    edgelist = res.load_resource('entailment-graph', 'edgelist')
    typemap = res.load_resource('entailment-graph', 'typemap')
    daganlevy = res.load_dataset('daganlevy', 'analysis')
    
    eg = EntailmentGraph(edgelist=edgelist, typemap=typemap)
    result = eg.run(daganlevy)
    print(result)


if __name__ == '__main__':
    main()