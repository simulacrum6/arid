import pandas as pd
import numpy as np

orig_edgelist = pd.read_csv('original-edgelist.txt', sep='\t-R?>\t')
orig_edgelist.columns = ['text', 'hypothesis']

# analysis edgelist
edgelist = orig_edgelist.copy()
edgelist = edgelist[['text', 'hypothesis']].replace('[<>]*', '', regex=True)
edgelist = edgelist[['text', 'hypothesis']].replace('[\.,:]\s', ' ', regex=True)
for column in ['text', 'hypothesis']:
    edgelist[column] = edgelist[column].str.split('::')
    edgelist[column] = [[x,pred,y] for pred,x,y in edgelist[column].values]

edgelist['entailment'] = [True] * len(edgelist)
edgelist.to_json('entailment-graph.json')

edgelist_tidy = edgelist.copy()
edgelist_tidy['tx'], edgelist_tidy['tpred'], edgelist_tidy['ty'] = list(zip(*([x,pred,y] for x,pred,y in edgelist.text)))
edgelist_tidy['hx'], edgelist_tidy['hpred'], edgelist_tidy['hy'] = list(zip(*([x,pred,y] for x,pred,y in edgelist.hypothesis)))
edgelist_tidy.text = edgelist_tidy['tx'] + ' ' + edgelist_tidy['tpred'] + ' ' + edgelist_tidy['ty']
edgelist_tidy.hypothesis = edgelist_tidy['hx'] + ' ' + edgelist_tidy['hpred'] + ' ' + edgelist_tidy['hy']
edgelist_tidy.to_csv('entailment-graph_tidy.csv')