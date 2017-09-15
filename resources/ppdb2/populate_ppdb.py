import pandas as pd
import sqlite3 as sql

# use phrasal-l, due to memory restrictions on laptop

def has_vp(string):
    if string.find('VP') != -1:
        return True
    else:
        return False

def to_sql(databases):
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
            chunk = chunk[[has_vp(entry) for entry in chunk['lhs'].values]]
            chunk[['phrase', 'paraphrase']].to_sql('paraphrases', con, if_exists='append')

con.close()

def to_csv(databases):
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
            usecols=['lhs', 'phrase', 'paraphrase', 'entailment']
        )
        
        for chunk in paraphrases:
            chunk = chunk[[entry in entailment_types for entry in chunk['entailment'].values]]
            chunk = chunk[[has_vp(entry) for entry in chunk['lhs'].values]]
            chunk[['phrase', 'paraphrase']].to_csv('ppdb.csv', index=False)

if __name__ == '__main__':
    databases = ['ppdb-2.0-l-phrasal']
    con = sql.connect('ppdb2.sqlite')
    entailment_types = ['ReverseEntailment', 'ForwardEntailment', 'Equivalence']
    
    to_csv(databases)
