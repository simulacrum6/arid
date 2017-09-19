import pandas as pd
import numpy as np
import re

###
# Berant 2011
###
orig_edgelist = pd.read_csv('original-edgelist.txt', sep='\t')
orig_edgelist.columns = ['text', 'edge', 'hypothesis']


edgelist = orig_edgelist.copy()
edgelist = edgelist[['text', 'hypothesis']].replace('[<>]*', '', regex=True)
edgelist = edgelist[['text', 'hypothesis']].replace('[\.,:]\s', ' ', regex=True)
edgelist['edge'] = orig_edgelist['edge']
for column in ['text', 'hypothesis']:
    edgelist[column] = edgelist[column].str.split('::')
    edgelist[column] = [[x,pred,y] for pred,x,y in edgelist[column].values]

edgelist_tidy = edgelist.copy()
edgelist_tidy['entailment'] = [True] * len(edgelist_tidy)
edgelist_tidy['tx'], edgelist_tidy['tpred'], edgelist_tidy['ty'] = list(zip(*([x,pred,y] for x,pred,y in edgelist.text)))
edgelist_tidy['hx'], edgelist_tidy['hpred'], edgelist_tidy['hy'] = list(zip(*([x,pred,y] for x,pred,y in edgelist.hypothesis)))
edgelist_tidy.text = edgelist_tidy['tx'] + ' ' + edgelist_tidy['tpred'] + ' ' + edgelist_tidy['ty']
edgelist_tidy.hypothesis = edgelist_tidy['hx'] + ' ' + edgelist_tidy['hpred'] + ' ' + edgelist_tidy['hy']
for _,row in edgelist_tidy.iterrows():
    if row['edge'] == '-R>':
        row['hx'], row['hy'] = row['hy'],row['hx']
edgelist_tidy.to_csv('berant_2011_tidy.csv')

edgelist_typed = edgelist_tidy[['text','hypothesis']].values
np.save('berant_2011_typed.npy', edgelist_typed)

edgelist_nc = []
for i,row in edgelist_tidy.iterrows():
    if row['edge'] == '-R>':
        entry = [['x', row['tpred'], 'y'], ['y', row['hpred'], 'x']]
    else:
        entry = [['x', row['tpred'], 'y'], ['x', row['hpred'], 'y']]
    edgelist_nc.append(entry)

np.save('berant_2011_mapped.npy', edgelist_nc)

##
# Berant Phd
##

def map_args(text, hypothesis):
    result = 0
    if text.endswith('@R@'):
        text = text.replace('@R@', '')
        result = result + 1
    if hypothesis.endswith('@R@'):
        hypothesis = hypothesis.replace('@R@', '')
        result = result + 1
    if result % 2 == 0:
        t = ' '.join(['x', text, 'y'])
        h = ' '.join(['x', hypothesis, 'y'])
    else:
        t = ' '.join(['x', text, 'y'])
        h = ' '.join(['y', hypothesis, 'x'])
    return [t,h]

# lambda = 0.05 

orig_edgelist = pd.read_csv('reverb_global_clsf_all_htl_lambda_0.05.txt', sep = '\t')
orig_edgelist.columns = ['text', 'hypothesis']

orig_edgelist['t'] = [pred.replace('@R@', '') for pred in orig_edgelist.text] 
orig_edgelist['h'] = [pred.replace('@R@', '') for pred in orig_edgelist.hypothesis]

np.save('berant_2010-0.05.npy', orig_edgelist[['t', 'h']].values)

# lambda = 0.05 mapped

orig_edgelist = pd.read_csv('reverb_global_clsf_all_htl_lambda_0.05.txt', sep = '\t')
orig_edgelist.columns = ['text', 'hypothesis']

mapped_edgelist = np.array([map_args(t,h) for t,h in orig_edgelist.values])
np.save('berant_2010-0.05_mapped.npy', mapped_edgelist)

# lambda = 0.1

orig_edgelist = pd.read_csv('reverb_global_clsf_all_tncf_lambda_0.1.txt', sep = '\t')
orig_edgelist.columns = ['text', 'hypothesis']

orig_edgelist['t'] = [pred.replace('@R@', '') for pred in orig_edgelist.text] 
orig_edgelist['h'] = [pred.replace('@R@', '') for pred in orig_edgelist.hypothesis]

np.save('berant_2010-0.1.npy', orig_edgelist[['t', 'h']].values)

orig_edgelist = pd.read_csv('reverb_global_clsf_all_tncf_lambda_0.1.txt', sep = '\t')
orig_edgelist.columns = ['text', 'hypothesis']

mapped_edgelist = np.array([map_args(t,h) for t,h in orig_edgelist.values])
np.save('berant_2010-0.1_mapped.npy', mapped_edgelist)
