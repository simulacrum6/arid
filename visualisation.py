import comparison as comp
import numpy as np
import utils.resources as res
import matplotlib.pyplot as plt
import os.path as path
from inference import Evaluator

# frequency by rank
def count_by_rank(datasets, plotname = 'cbr.png'):
    count_by_rank = plt.figure('cbr')
    cbr = count_by_rank.add_subplot(111)
    lines = []
    for name, dataset in datasets.items():
        xy_data = comp.predicates(dataset).value_counts().reset_index(drop=True)
        line, = cbr.loglog(xy_data, label = name)
        lines.append(line)
    cbr.legend(handles = lines)
    cbr.set_title('Frequency count by Rank')
    cbr.set_xlabel('Rank')
    cbr.set_ylabel('Frequency in Dataset')
    plt.figure('cbr')
    plt.tight_layout()
    plt.savefig(path.join(
            OUTPATH,
            plotname))
    plt.clf()

def frequency_density_distribution(datasets, plotname = 'fdd.png'):
    frequency_density_distribution = plt.figure('fdd')
    fdd = frequency_density_distribution.add_subplot(111)
    lines = []
    for name, dataset in datasets.items():
        xy_data = comp.predicates(dataset).value_counts().to_frame().iloc[:,0].value_counts().sort_index()
        line, = fdd.step(xy_data.index, xy_data.values, label=name, where='post')
        lines.append(line)
    fdd.legend(handles = lines)
    fdd.set_title('Predicate frequency distribtuion')
    fdd.set_xlabel('Occurrence Frequency of Predicates')
    fdd.set_ylabel('Count in Dataset')
    fdd.set_yscale('log')
    fdd.set_xscale('log')
    plt.figure('fdd')
    plt.tight_layout()
    plt.savefig(path.join(
            OUTPATH,
            plotname))
    plt.clf()

def grouped_barplot(datasets, plotname):
    fig = plt.figure('bar')
    ax = fig.add_subplot(111)
      
    N = len(datasets.keys()) +1
    ind = np.arange(N)
    width = 0.15
    
    bars = []
    i = 0
    for name, dataset in datasets.items():
        values = [
            len(comp.unique_predicates(dataset)), 
            len(comp.unique_attributes(dataset)),
            len(comp.unique_templates(dataset))
        ]
        bar = ax.bar(ind + width*i, values, width, label = name)
        bars.append(bar)
        i = i + 1
    
    ax.set_title('Count of unique instances')
    ax.set_ylabel('Count')
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(['Predicates', 'Attributes', 'Propositions'])
    ax.legend(handles = bars)
    
    plt.figure('bar')
    plt.tight_layout()
    plt.savefig(path.join(
        OUTPATH,
        plotname
    ))
    plt.clf()

def plot_prec_rec(results, plotname = 'rec-prec_dl-z.png', classifiers = ['Lemma Baseline', 'Entailment Graph', 'Relation Embeddings']):
    fig = plt.figure('res')
    ax = fig.add_subplot(111)
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    lines = []
    for name, result in results.items():
        predictions = result[classifiers].T.values
        gold = result['Gold']
        
        auc = Evaluator.auc(gold, predictions)
        rec, prec, thresh = Evaluator.precision_recall_curve(gold, predictions)
        
        line, = ax.step(rec[1:-2], prec[1:-2], where = 'post', label = '{0} ({1}). AUC:{2:.3f}'.format(name, 'Full', auc))
        lines.append(line)
            
    ax.legend(handles = lines)
    plt.figure('res')
    plt.tight_layout()
    plt.xlim([0,1.0])
    plt.ylim([0,1.05])
    plt.savefig(path.join(
        OUTPATH,
        plotname
    ))
    plt.clf()


def make_plots():
    OUTPATH = path.join(res.output, 'figures')

    datasets = {
        'Levy & Dagan': res.load_dataset('daganlevy', 'tidy'),
        'Zeichner et. al': res.load_dataset('zeichner', 'tidy')
    }
    results = {
        'Levy & Dagan': res.load_result('daganlevy'),
        'Zeichner et. al': res.load_result('zeichner')   
    }
    count_by_rank(datasets, plotname = 'cbr_dl-z.png')
    frequency_density_distribution(datasets, plotname = 'fdd_dl-z.png')
    grouped_barplot(datasets, plotname = 'pa-freq_dl-z.png')
    plot_prec_rec(results)

if __name__ == '__main__':
    make_plots()

