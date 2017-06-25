import pandas as pd
import numpy as np

daganlevy = pd.read_csv(
    '.\\original-datasets\\daganlevy.txt', 
    sep='\t',
    header=None,
    names=['text', 'hypothesis', 'entailment'])

text_headers = ['text_x', 'text_predicate', 'text_y']
hypothesis_headers = ['hypothesis_x', 'hypothesis_predicate', 'hypothesis_y']

text = pd.DataFrame(
    (text.split(', ') for text in daganlevy.text),
    columns=text_headers)
hypothesis = pd.DataFrame(
    (text.split(', ') for text in daganlevy.hypothesis),
    columns=hypothesis_headers)
daganlevy_tidy = pd.concat(
    [text, hypothesis, daganlevy.entailment],
    axis=1)
daganlevy_tidy.to_csv('.\\working-datasets\\daganlevy-tidy.csv')

annoToVal = {
    'y': True,
    'n': False
}

daganlevy_analysis = daganlevy.copy()
daganlevy_analysis.text = [text.split(', ') for text in daganlevy.text]
daganlevy_analysis.hypothesis = [text.split(', ') for text in daganlevy.hypothesis]
daganlevy_analysis.annotation = [annoToVal[annotation] for annotation in daganlevy.annotation]
daganlevy_analysis.to_csv('.\\working-datasets\\daganlevy.csv')


text_headers = ['text_x', 'text_predicate', 'text_y']
hypothesis_headers = ['hypothesis_x', 'hypothesis_predicate', 'hypothesis_y']

zeichner_entailing = pd.read_csv(
    '.\\original-datasets\\zeichner_entailingAnnotations.txt', 
    sep='\t')
zeichner_nonEntailing = pd.read_csv(
    '.\\original-datasets\\zeichner_nonEntailingAnnotations.txt', 
    sep='\t')
zeichner = pd.concat([zeichner_entailing, zeichner_nonEntailing]).reset_index(drop=True)

# TODO: Fix broken entries
t_x, t_y = zip(*(text.split(' ' + rule + ' ') if len(text.split(' ' + rule + ' ')) == 2 else ['NaN', 'NaN'] for text,rule in zip(zeichner.lhs, zeichner.rule_lhs)))
h_x, h_y = zip(*(text.split(' ' + rule + ' ') if len(text.split(' ' + rule + ' ')) == 2 else ['NaN', 'NaN'] for text,rule in zip(zeichner.rhs, zeichner.rule_rhs)))

z_anno = {
    'Yes': 'y',
    'yes': 'y',
    'No': 'n',
    'no': 'n'
}

text = pd.DataFrame(
    list(zip(t_x, zeichner.rule_lhs, t_y)),
    columns=text_headers)
hypothesis = pd.DataFrame(
    list(zip(h_x, zeichner.rule_rhs, h_y)),
    columns=hypothesis_headers)
annotation = pd.DataFrame(
    (z_anno[annotation] for annotation in zeichner.judgment),
    columns=['entailment'])
zeichner_tidy = pd.concat(
    [text, hypothesis, annotation],
    axis=1)
zeichner_tidy.to_csv('.\\working-datasets\\zeichner-tidy.csv')


text = zip(zeichner_tidy.text_x, zeichner_tidy.text_predicate, zeichner_tidy.text_y)
hypothesis = zip(zeichner_tidy.hypothesis_x, zeichner_tidy.hypothesis_predicate, zeichner_tidy.hypothesis_y)

zeichner_working = pd.concat(
    [zeichner.lhs, zeichner.rhs, zeichner_tidy.entailment],
    axis=1)
zeichner_working.columns = ['text', 'hypothesis', 'entailment']
zeichner_working.text = [[x, pred, y] for x, pred, y in text]
zeichner_working.hypothesis = [[x, pred, y] for x, pred, y in hypothesis]
zeichner_working.entailment = [annoToVal[annotation] for annotation in zeichner_working.entailment]
zeichner_working.to_csv('.\\working-datasets\\zeichner.csv')