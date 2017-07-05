import numpy as np
 
 # TODO: allow non exact matching mode by matching substrings in x, pred, y 

class EntailmentGraph:
    def __init__(self, edgelist, typemap):
        self.typemap = typemap
        self.edgelist = np.transpose(edgelist)
    
    def run(self, dataset):
        self.type_attributes(dataset)
        return np.array([self.evaluate(text, hypothesis) for text,hypothesis in dataset])
    
    def type_attributes(self, dataset):
        dataset[0] = np.array([self.type(x), pred, self.type(y)] for x,pred,y in dataset[0])
        dataset[1] = np.array([self.type(x), pred, self.type(y)] for x,pred,y in dataset[1])
    
    def type(self, string):
        if self.typemap[string]:
            return self.typemap[string]
        else:
            return string
    
    # exact matching
    def evaluate(self, text, hypothesis):
        if (text, hypothesis) in self.edgelist:
            return True
        else:
            return False


def main():
    #temporary solution
    import sys, os
    sys.path.append('C:\\Users\\Nev\\Projects')
    from arid.utils import resources as res
    
    edgelist = res.load_resource('entailment-graph', 'edgelist')
    typemap = res.load_resource('entailment-graph', 'typemap')
    daganlevy = res.load_dataset('daganlevy', 'analysis')
    
    eg = EntailmentGraph(edgelist=edgelist, typemap=typemap)
    result = eg.run(daganlevy)
    print(result)


if __name__ == '__main__':
    main()