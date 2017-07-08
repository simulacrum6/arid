import pandas as pd
import sqlite3 as sql

filename = 'ppdb-2.0-s-lexical'

data = pd.read_csv(filename, 
    sep=' \|\|\| ', 
    iterator=True, 
    chunksize=50000,
    names=['lhs', 'phrase', 'paraphrase', 'feature_value', 'alignment', 'entailment'])

con = sql.connect('ppdb3.sqlite')

for chunk in data:
    chunk.to_sql('paraphrases', con, if_exists='replace')

print(pd.read_sql_query('SELECT phrase, paraphrase FROM paraphrases WHERE phrase = "provided"', con))

con.close()

#pd.read_sql_query(
#    'SELECT paraphrases.entailment ' +
#    'FROM  paraphrases ' +
#    'JOIN daganlevy ' + 
#    'ON daganlevy.tpred = paraphrases.phrase ' + 
#    'AND daganlevy.hpred = paraphrases.paraphrase ', con)