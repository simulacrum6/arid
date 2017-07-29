import pandas as pd
import numpy as np
import os.path as path
import sys
# Temporary solution
sys.path.append('C:\\Users\\Nev\\Projects\\')
import arid.utils.resources as res
import arid.utils.qa_utils as qa_utils

# map annotation Strings to boolean
annoToVal = {
    'y': True,
    'yes': True,
    'Yes': True,
    'n': False,
    'no': False,
    'No': False            
}

COLUMN_NAMES = ['text', 'hypothesis', 'entailment']
TEXT_HEADERS = ['tx', 'tpred', 'ty']
HYPOTHESIS_HEADERS = ['hx', 'hpred', 'hy']
INPUT_PATH = path.join(res.resources, 'original-datasets')
OUTPUT_PATH = path.join(res.resources, 'datasets')


###
# Create Dagan & Levy Datasets
###
daganlevy = pd.read_csv(
    path.join(INPUT_PATH, 'daganlevy.txt'), 
    sep='\t',
    header=None,
    names=COLUMN_NAMES)


# Create tidy dataset
text = pd.DataFrame(
    daganlevy.text.replace(',', '', regex=True))
text_split = pd.DataFrame(
    (text.split(', ') for text in daganlevy.text),
    columns=TEXT_HEADERS)
hypothesis = pd.DataFrame(
    daganlevy.hypothesis.replace(',', '', regex=True))
hypothesis_split = pd.DataFrame(
    (text.split(', ') for text in daganlevy.hypothesis),
    columns=HYPOTHESIS_HEADERS)
annotation = pd.DataFrame(
    (annoToVal[annotation] for annotation in daganlevy.entailment),
    columns=['entailment'])

daganlevy_tidy = pd.concat(
    [text, hypothesis, annotation, text_split, hypothesis_split],
    axis=1)

# Create analysis dataset
daganlevy_analysis = daganlevy.copy()
daganlevy_analysis.text = [text.split(', ') for text in daganlevy.text]
daganlevy_analysis.hypothesis = [text.split(', ') for text in daganlevy.hypothesis]
daganlevy_analysis.entailment = [annoToVal[annotation] for annotation in daganlevy.entailment]


# Create lemmatised dataset

daganlevy_lemmatised = daganlevy_tidy.copy()
text_lemmatised = [' '.join(qa_utils.get_lemmas(entry)) for entry in daganlevy_tidy.tpred]
hypothesis_lemmatised = [' '.join(qa_utils.get_lemmas(entry)) for entry in daganlevy_tidy.hpred]

daganlevy_lemmatised.tpred = text_lemmatised
daganlevy_lemmatised.hpred = hypothesis_lemmatised
daganlevy_lemmatised.text = [
    entry.replace(raw, lemma) 
    for entry, raw, lemma in zip(
        daganlevy_lemmatised.text, 
        daganlevy_tidy.tpred, 
        daganlevy_lemmatised.tpred)]
daganlevy_lemmatised.hypothesis = [
    entry.replace(raw, lemma) 
    for entry, raw, lemma in zip(
        daganlevy_lemmatised.hypothesis, 
        daganlevy_tidy.hpred, 
        daganlevy_lemmatised.hpred)]

# Write to file
daganlevy_tidy.to_csv(path.join(OUTPUT_PATH, 'daganlevy-tidy.csv'))
daganlevy_lemmatised.to_csv(path.join(OUTPUT_PATH, 'daganlevy-tidy_lemmatised.csv'))
daganlevy_analysis.to_json(path.join(OUTPUT_PATH, 'daganlevy.json'))

###
# Create Zeichner Datasets
###
zeichner_entailing = pd.read_csv(
    path.join(INPUT_PATH, 'zeichner_entailingAnnotations.txt'), 
    sep='\t')
zeichner_nonEntailing = pd.read_csv(
    path.join(INPUT_PATH, 'zeichner_nonEntailingAnnotations.txt'), 
    sep='\t')
zeichner = pd.concat([zeichner_entailing, zeichner_nonEntailing]).reset_index(drop=True)


# Create tidy dataset

# extract attributes from templates, filtering for reversal markers.
# NOTE: Mention reversal markers in dataset description
tx, ty = zip(*(text.split(' ' + rule.replace('@R@', '') + ' ') for text,rule in zip(zeichner.lhs, zeichner.rule_lhs)))
hx, hy = zip(*(text.split(' ' + rule.replace('@R@', '') + ' ') for text,rule in zip(zeichner.rhs, zeichner.rule_rhs)))

text = pd.DataFrame(
    zeichner.lhs.values, 
    columns=['text'])
text_split = pd.DataFrame(
    list(zip(tx, zeichner.rule_lhs, ty)),
    columns=TEXT_HEADERS)
hypothesis = pd.DataFrame(
    zeichner.rhs.values,
    columns=['hypothesis'])
hypothesis_split = pd.DataFrame(
    list(zip(hx, zeichner.rule_rhs, hy)),
    columns=HYPOTHESIS_HEADERS)
annotation = pd.DataFrame(
    (annoToVal[annotation] for annotation in zeichner.judgment),
    columns=['entailment'])
zeichner_tidy = pd.concat(
    [text, hypothesis, annotation, text_split, hypothesis_split],
    axis=1)

# weed out invalid entries. Obsolete now. 
valid = zeichner_tidy.hx != 'NaN'
nans = zeichner_tidy.hx == 'NaN'

zeichner_dirty = zeichner_tidy[nans]
zeichner_tidy = zeichner_tidy[valid]

# Create analysis dataset
text = zip(zeichner_tidy.tx, zeichner_tidy.tpred, zeichner_tidy.ty)
hypothesis = zip(zeichner_tidy.hx, zeichner_tidy.hpred, zeichner_tidy.hy)

zeichner_analysis = pd.concat(
    [zeichner_tidy.tpred, zeichner_tidy.hpred, zeichner_tidy.entailment],
    axis=1)
zeichner_analysis.columns = COLUMN_NAMES
zeichner_analysis.text = [[x, pred, y] for x, pred, y in text]
zeichner_analysis.hypothesis = [[x, pred, y] for x, pred, y in hypothesis]

# Write to file
zeichner.to_csv(path.join(INPUT_PATH, 'zeichner.txt'))
zeichner_tidy.to_csv(path.join(OUTPUT_PATH, 'zeichner-tidy.csv'))
zeichner_dirty.to_csv(path.join(OUTPUT_PATH, 'zeichner-dirty.csv'))
zeichner_analysis.to_json(path.join(OUTPUT_PATH, 'zeichner.json'))