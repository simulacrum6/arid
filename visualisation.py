import comparison as comp
import numpy as np
import utils.resources as res
import matplotlib.pyplot as plt
import os.path as path


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
    #plt.show()

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
    #plt.show()

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
            len(comp.unique_templates(dataset))]
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
    #plt.show()


def make_plots():
    OUTPATH = path.join(res.output, 'figures')
    
    datasets = {
        'Levy & Dagan': res.load_dataset('daganlevy', 'tidy'),
        'Zeichner et. al': res.load_dataset('zeichner', 'tidy')
        }
    count_by_rank(datasets, plotname = 'cbr_dl-z.png')
    frequency_density_distribution(datasets, plotname = 'fdd_dl-z.png')
    grouped_barplot(datasets, plotname = 'pa-freq_dl-z.png')


if __name__ == '__main__':
    make_plots()

