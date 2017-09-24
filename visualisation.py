import comparison as comp
import pandas as pd
import numpy as np
import utils.resources as res
import matplotlib.pyplot as plt
import os.path as path
from utils.evaluation import Evaluator, get_sample
import sklearn.metrics as skm
import matplotlib.patches as mp
import matplotlib.lines as ml

OUTPATH = path.join(res.output, 'figures')
colors = {
    'Levy & Dagan': 'C0',
    'Zeichner et al.': 'C1'
}


def count_by_rank(datasets, plotname = 'cbr.png'):
    count_by_rank = plt.figure(plotname)
    cbr = count_by_rank.add_subplot(111)
    legend = [
        ml.Line2D([-1], [-1], color='black', label='All', linestyle='-'),
        ml.Line2D([-1], [-1], color='black', label='Text', linestyle=':'),
        ml.Line2D([-1], [-1], color='black', label='Hypothesis', linestyle='--')
    ]
    for name, dataset in datasets.items():
        legend.append(mp.Patch(label=name, color=colors[name]))
        all_preds = comp.predicates(dataset).value_counts().reset_index(drop=True)
        t_preds = dataset.tpred.value_counts().reset_index(drop=True)
        h_preds = dataset.hpred.value_counts().reset_index(drop=True)
        cbr.loglog(all_preds, color=colors[name], linestyle='-', linewidth=2)
        cbr.loglog(t_preds, color=colors[name], linestyle=':', alpha=0.75)
        cbr.loglog(h_preds, color=colors[name], linestyle='--', alpha=0.75)
    
    cbr.legend(handles = legend)
    cbr.set_title('Frequency count by Rank')
    cbr.set_xlabel('Rank')
    cbr.set_ylabel('Frequency in Dataset')
    plt.figure(plotname)
    plt.tight_layout()
    plt.savefig(path.join(OUTPATH, plotname))

def frequency_density_distribution(datasets, plotname = 'fdd.png'):
    frequency_density_distribution = plt.figure('fdd')
    fdd = frequency_density_distribution.add_subplot(111)
    lines = []
    for name, dataset in datasets.items():
        xy_data = comp.predicates(dataset).value_counts().to_frame().iloc[:,0].value_counts().sort_index()
        line, = fdd.step(xy_data.index, xy_data.values, label=name, where='post', color=colors[name])
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
        bar = ax.bar(ind + width*i, values, width, label = name, color=colors[name])
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

def plot_prec_rec(results, ensembles, plotname = 'rec-prec_dl-z.png'):
    
    fig = plt.figure(plotname)
    n = len(ensembles.keys())*10
    i = 101 + n
    for ensemblename, classifiers in ensembles.items():
        ax = fig.add_subplot(i)
        ax.set_title(ensemblename)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim(0,1.05)
        ax.set_ylim(0,1.0)
        lines = []
        for name, result in results.items():
            predictions = result[classifiers].T.values
            gold = result['Gold'].values
            auc = Evaluator.auc(gold, predictions)
            prec, rec, _ = Evaluator.precision_recall_curve(gold, predictions)
            
            line, = ax.step(rec, prec, where = 'post', label = '{0} (AUC={1:.3f})'.format(name, auc), color=colors[name], alpha=0.8)
            lines.append(line)
            ax.fill_between(rec, prec, step='post', alpha=0.25, color=colors[name])
            
            if ensemblename == list(ensembles.keys())[0]:
                ax.plot(rec[-2], prec[-2], marker = 'x', color='black')
                ax.text(rec[-2], prec[-2], '({0:.2f},{1:.2f})'.format(rec[-2], prec[-2]))
                
        ax.legend(handles = lines)
        i = i + 1
    plt.figure(plotname)
    #plt.tight_layout()
    plt.savefig(path.join(
        OUTPATH,
        plotname
    ))

def plot_points(results, points, plotname='ind_dl-z.png'):
    fig = plt.figure('ind')
    ax = fig.add_subplot(111)
    ax.set_title('Precision & Recall Values for individual Methods ')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim(0,1.05)
    ax.set_xlim(0,0.2)
    legend = [ml.Line2D([0], [0], color='black', label=methodname, marker=marker, linestyle='None') for methodname,marker in points.items()]
    for name, result in results.items():
        legend.append(mp.Patch(label=name, color=colors[name]))
        gold = result['Gold']
        for method, marker in points.items():
            predictions = result[method].values
            precision = skm.accuracy_score(gold, predictions)
            recall = skm.recall_score(gold, predictions)
            point, = ax.plot(recall, precision, marker = marker, color=colors[name])
            #ax.text(recall, precision, '({0:.2f},{1:.2f})'.format(recall, precision))
    ax.legend(
        handles=legend,
        loc='lower right')
    plt.figure('ind')
    plt.tight_layout()
    plt.savefig(path.join(
        OUTPATH,
        plotname
    ))

def plot_mean_aucs(plotname='mean-aucs.png'):
    aucs = pd.read_csv(path.join(res.output, 'mean_aucs.csv'))
    daganlevy = aucs[aucs['Dataset'] == 'daganlevy']
    daganlevy = daganlevy[daganlevy['Ensemble'] == 'Combined Methods'][['Positive Percentage', 'MAP']]
    zeichner = aucs[aucs['Dataset'] == 'zeichner']
    zeichner = zeichner[zeichner['Ensemble'] == 'Combined Methods'][['Positive Percentage', 'MAP']]
    datasets = {'Levy & Dagan': daganlevy, 'Zeichner et al.': zeichner}
    
    fig = plt.figure(plotname)
    ax = fig.add_subplot(111)
    ax.set_title('Mean AUC Values for different Positive-Rates')
    ax.set_xlabel('Positive Rate')
    ax.set_ylabel('Mean AUC')
    ax.set_ylim(0,1.05)
    ax.set_xlim(0.15,0.85)
    legend = []
    
    for name, dataset in datasets.items():
        line, = ax.plot(dataset['Positive Percentage'], dataset['MAP'], label=name, color=colors[name], linestyle='None', marker='.')
        legend.append(line)
    
    ax.legend(handles=legend)
    plt.figure(plotname)
    plt.tight_layout()
    plt.savefig(path.join(OUTPATH,plotname))

def daganlevy_reproduction(plotname='dlr.png'):
    result = res.load_result('daganlevy')
    gold = result['Gold'].values
    fig = plt.figure('dlr')
    ax = fig.add_subplot(111)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim(0.5,1.0)
    ax.set_xlim(0,0.5)
    points = {
        'Lemma': {
            'values': ['Lemma Baseline'],
            'marker': 'p',
            'color': 'black'
        },
        'PPDB': {
            'values': ['Lemma Baseline', 'PPDB'],
            'marker': 'o',
            'color': '#ff006e'
        },
        'Entailment Graph': {
            'values': ['Lemma Baseline', 'Entailment Graph'],
            'marker': 's',
            'color': 'blue'
        },
        'All Rules': {
            'values': ['Lemma Baseline', 'Entailment Graph', 'PPDB'],
            'marker': '*',
            'color': '#ff006e'
        }
    }
    legend = []
    for name, props in points.items():
        predictions = result[props['values']].T.values
        prediction = Evaluator.aggregate(predictions, max)
        precision = skm.precision_score(gold, prediction)
        recall = skm.recall_score(gold, prediction)
        line, = ax.plot([recall], [precision], marker = props['marker'], markersize=10, color = props['color'], label=name, linestyle='None')
        legend.append(line)
    predictions = result[['Lemma Baseline', 'Relation Embeddings']].T.values
    prediction = Evaluator.aggregate(predictions, max)
    prec, rec, thresh = skm.precision_recall_curve(gold, prediction)
    line, = ax.plot(rec[1:-1], prec[1:-1], color='green', linestyle='--', linewidth=1, label='Relation Embs')
    legend.append(line)
    plt.figure('dlr')
    plt.legend(handles = legend)
    plt.tight_layout()
    plt.savefig(path.join(
        OUTPATH,
        plotname
    ))
    plt.show()



def make_plots():
    datasets = {
        'Levy & Dagan': res.load_dataset('daganlevy', 'tidy'),
        'Zeichner et al.': res.load_dataset('zeichner', 'tidy')
    }
    results = {
        'Levy & Dagan': res.load_result('daganlevy'),
        'Zeichner et al.': res.load_result('zeichner')   
    }
    samples = {
        'Levy & Dagan': res.load_result('daganlevy'),
        'Zeichner et al.': res.load_result('zeichner')   
    }
    ensembles = {
        'Combined Methods': [
            'Lemma Baseline', 
            'Entailment Graph', 
            #'Berant (2011)',
            'PPDB', 
            'Relation Embeddings'
        ],
        'Embeddings only': ['Relation Embeddings']
    }
    points = {
        'Lemma Baseline': 'x',
        'Entailment Graph': '^',
        #'Berant (2011)': 'v',
        'PPDB': 'o'
    }
    count_by_rank(datasets, plotname = 'cbr_dl-z.png')
    frequency_density_distribution(datasets, plotname = 'fdd_dl-z.png')
    grouped_barplot(datasets, plotname = 'pa-freq_dl-z.png')
    plot_prec_rec(results, ensembles)
    plot_prec_rec(samples, ensembles, rec-prec_samples.png)
    plot_points(results, points)
    plot_mean_aucs()
    plt.show()

if __name__ == '__main__':
    make_plots()

