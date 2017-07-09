import pandas as pd
import sqlite3 as sql

# use phrasal-l, due to memory restrictions on laptop
databases = ['ppdb-2.0-xxxl-lexical', 'ppdb-2.0-l-phrasal']
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
        ])

    for chunk in paraphrases:
        chunk.to_sql('paraphrases', con, if_exists='append')

con.close()