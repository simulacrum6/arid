import pandas as pd

daganlevy = pd.read_csv(
    '.\\original-datasets\\daganlevy.txt', 
    sep='\t',
    header=None,
    names=['text', 'hypothesis', 'annotation'])

text = pd.DataFrame(
    (text.split(', ') for text in daganlevy.text),
    columns=['text_x', 'text_predicate', 'text_y'])
hypothesis = pd.DataFrame(
    (text.split(', ') for text in daganlevy.hypothesis),
    columns=['hypothesis_x', 'hypothesis_predicate', 'hypothesis_y'])
daganlevy_tidy = pd.concat(
    [text, hypothesis, daganlevy.annotation],
    axis=1)
daganlevy_tidy.to_csv('.\\working-datasets\\daganlevy-tidy.csv')