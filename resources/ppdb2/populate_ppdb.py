import pandas as pd
import sqlite3 as sql

# use phrasal-l, due to memory restrictions on laptop

def has_verb(string):
    if string.find('V') != -1:
        return True
    else:
        return False

def to_sql(databases):
    con = sql.connect('ppdb2.sqlite')
    for db in databases:
        paraphrases = pd.read_csv(
            db,
            sep=' \|\|\| ',
            iterator=True,
            chunksize=50000,
            names=[
                'lhs', 'phrase', 'paraphrase', 'feature_value', 'alignment',
                'entailment'
            ],
            usecols=['lhs', 'phrase', 'paraphrase', 'entailment'])

        for chunk in paraphrases:
            chunk = chunk[[entry in entailment_types for entry in chunk['entailment'].values]]
            chunk = chunk[[has_verb(entry) for entry in chunk['lhs'].values]]
            chunk[['phrase', 'paraphrase']].to_sql('paraphrases', con, if_exists='append')

con.close()

def to_csv(databases):
    for db in databases:
        paraphrases = pd.read_csv(
            db,
            sep=' \|\|\| ',
            iterator=True,
            chunksize=1000000,
            names=[
                'lhs', 'phrase', 'paraphrase', 'feature_value', 'alignment',
                'entailment'
            ],
            usecols=['lhs', 'phrase', 'paraphrase', 'entailment']
        )
        
        for chunk in paraphrases:
            chunk = chunk[[entry in entailment_types for entry in chunk['entailment'].values]]
            chunk = chunk[[has_verb(entry) for entry in chunk['lhs'].values]]
            chunk[['phrase', 'paraphrase']].to_csv('ppdb2.csv', index=False, mode = 'a')

if __name__ == '__main__':
    databases = ['ppdb-2.0-l-phrasal', 'ppdb-2.0-xxxl-lexical']
    entailment_types = ['ReverseEntailment', 'ForwardEntailment', 'Equivalence']
    small = ['ppdb-2.0-s-lexical']
    to_csv(databases)
