import pandas as pd
import numpy as np

# Maps annotation Strings to En
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



### Create Dagan & Levy Datasets ###
daganlevy = pd.read_csv(
    '.\\original-datasets\\daganlevy.txt', 
    sep='\t',
    header=None,
    names=['text', 'hypothesis', 'entailment'])


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
daganlevy_tidy.to_csv('.\\datasets\\daganlevy-tidy.csv')


# Create analysis dataset
daganlevy_analysis = daganlevy.copy()
daganlevy_analysis.text = [text.split(', ') for text in daganlevy.text]
daganlevy_analysis.hypothesis = [text.split(', ') for text in daganlevy.hypothesis]
daganlevy_analysis.entailment = [annoToVal[annotation] for annotation in daganlevy.entailment]
daganlevy_analysis.to_json('.\\datasets\\daganlevy.json')



### Create Zeichner Datasets
zeichner_entailing = pd.read_csv(
    '.\\original-datasets\\zeichner_entailingAnnotations.txt', 
    sep='\t')
zeichner_nonEntailing = pd.read_csv(
    '.\\original-datasets\\zeichner_nonEntailingAnnotations.txt', 
    sep='\t')
zeichner = pd.concat([zeichner_entailing, zeichner_nonEntailing]).reset_index(drop=True)


# Create tidy dataset
# TODO: Fix broken entries
tx, ty = zip(*(text.split(' ' + rule + ' ') if len(text.split(' ' + rule + ' ')) == 2 else ['NaN', 'NaN'] for text,rule in zip(zeichner.lhs, zeichner.rule_lhs)))
hx, hy = zip(*(text.split(' ' + rule + ' ') if len(text.split(' ' + rule + ' ')) == 2 else ['NaN', 'NaN'] for text,rule in zip(zeichner.rhs, zeichner.rule_rhs)))

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

valid = zeichner_tidy.hx != 'NaN'
nans = zeichner_tidy.hx == 'NaN'

zeichner_dirty = zeichner_tidy[nans]
zeichner_tidy = zeichner_tidy[valid]

zeichner_tidy.to_csv('.\\datasets\\zeichner-tidy.csv')
zeichner_dirty.to_csv('.\\datasets\\zeichner-dirty.csv')

# Create analysis dataset
text = zip(zeichner_tidy.tx, zeichner_tidy.tpred, zeichner_tidy.ty)
hypothesis = zip(zeichner_tidy.hx, zeichner_tidy.hpred, zeichner_tidy.hy)

zeichner_analysis = pd.concat(
    [zeichner_tidy.tpred, zeichner_tidy.hpred, zeichner_tidy.entailment],
    axis=1)
zeichner_analysis.columns = COLUMN_NAMES
zeichner_analysis.text = [[x, pred, y] for x, pred, y in text]
zeichner_analysis.hypothesis = [[x, pred, y] for x, pred, y in hypothesis]
zeichner_analysis.to_json('.\\datasets\\zeichner.json')