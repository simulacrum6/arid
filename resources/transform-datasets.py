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
TEXT_HEADERS = ['text_x', 'text_predicate', 'text_y']
HYPOTHESIS_HEADERS = ['hypothesis_x', 'hypothesis_predicate', 'hypothesis_y']



### Create Dagan & Levy Datasets ###
daganlevy = pd.read_csv(
    '.\\original-datasets\\daganlevy.txt', 
    sep='\t',
    header=None,
    names=['text', 'hypothesis', 'entailment'])


# Create tidy dataset
text = pd.DataFrame(
    (text.split(', ') for text in daganlevy.text),
    columns=TEXT_HEADERS)
hypothesis = pd.DataFrame(
    (text.split(', ') for text in daganlevy.hypothesis),
    columns=HYPOTHESIS_HEADERS)
annotation = pd.DataFrame(
    (annoToVal[annotation] for annotation in daganlevy.entailment),
    columns=['entailment'])

daganlevy_tidy = pd.concat(
    [text, hypothesis, annotation],
    axis=1)
daganlevy_tidy.to_csv('.\\working-datasets\\daganlevy-tidy.csv')


# Create analysis dataset
daganlevy_analysis = daganlevy.copy()
daganlevy_analysis.text = [text.split(', ') for text in daganlevy.text]
daganlevy_analysis.hypothesis = [text.split(', ') for text in daganlevy.hypothesis]
daganlevy_analysis.entailment = [annoToVal[annotation] for annotation in daganlevy.entailment]
daganlevy_analysis.to_json('.\\working-datasets\\daganlevy.json')



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
t_x, t_y = zip(*(text.split(' ' + rule + ' ') if len(text.split(' ' + rule + ' ')) == 2 else ['NaN', 'NaN'] for text,rule in zip(zeichner.lhs, zeichner.rule_lhs)))
h_x, h_y = zip(*(text.split(' ' + rule + ' ') if len(text.split(' ' + rule + ' ')) == 2 else ['NaN', 'NaN'] for text,rule in zip(zeichner.rhs, zeichner.rule_rhs)))

text = pd.DataFrame(
    list(zip(t_x, zeichner.rule_lhs, t_y)),
    columns=TEXT_HEADERS)
hypothesis = pd.DataFrame(
    list(zip(h_x, zeichner.rule_rhs, h_y)),
    columns=HYPOTHESIS_HEADERS)
annotation = pd.DataFrame(
    (annoToVal[annotation] for annotation in zeichner.judgment),
    columns=['entailment'])
zeichner_tidy = pd.concat(
    [text, hypothesis, annotation],
    axis=1)

valid = zeichner_tidy.hypothesis_x != 'NaN'
nans = zeichner_tidy.hypothesis_x == 'NaN'

zeichner_dirty = zeichner_tidy[nans]
zeichner_tidy = zeichner_tidy[valid]

zeichner_tidy.to_csv('.\\working-datasets\\zeichner-tidy.csv')
zeichner_dirty.to_csv('.\\working-datasets\\zeichner-dirty.csv')

# Create analysis dataset
text = zip(zeichner_tidy.text_x, zeichner_tidy.text_predicate, zeichner_tidy.text_y)
hypothesis = zip(zeichner_tidy.hypothesis_x, zeichner_tidy.hypothesis_predicate, zeichner_tidy.hypothesis_y)

zeichner_analysis = pd.concat(
    [zeichner_tidy.text_predicate, zeichner_tidy.hypothesis_predicate, zeichner_tidy.entailment],
    axis=1)
zeichner_analysis.columns = COLUMN_NAMES
zeichner_analysis.text = [[x, pred, y] for x, pred, y in text]
zeichner_analysis.hypothesis = [[x, pred, y] for x, pred, y in hypothesis]
zeichner_analysis.to_json('.\\working-datasets\\zeichner.json')