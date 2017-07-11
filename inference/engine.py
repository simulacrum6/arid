import sklearn.metrics as skm
import numpy as np

class ClassificationEngine:
    """Engine for running a set of relation inference classifiers over a dataset.

    Methods:
        run(classifiers, dataset) -- Runs a list of classifiers over dataset
        
        Parameters:
            classifiers -- Listlike of classifiers as arid.inference.classifiers.Classifier
            dataset -- Dataset to be classified as numpy.ndarray (2,3). Should contain Text and Hypothesis and X attribute, predicate, Y attribute for each
    
        Returns:
            A list of numpy.ndarrays (shape=(len(classifiers),len(dataset)) dtype='float'). 
            The list contains one array for each classifier. Each array contains either True/False for each entry in the dataset.
    """

    def __init__(self, classifiers):
        self.classifiers = classifiers
    
    def run(self, dataset):
        """Run a list of classifiers over dataset. See help(ClassificationEngine) for details"""
        return [cf.run(dataset) for cf in self.classifiers]



class Evaluator:
    def aggregate(self, predictions, function):
        """Aggregates each sublist of predictions, using the specified function"""
        return [function(sublist) for sublist in np.transpose(predictions)]
    
    def precision_recall(self, gold, prediction):
        prediction = self.aggregate(prediction, max)
        return skm.precision_recall_curve(gold, prediction)
    
    def auc(self, gold, prediction):
        precision, recall, _ = self.precision_recall(gold, prediction)
        return skm.auc(recall, precision)
    

def main():
    from random import random
    import numpy as np
    import sklearn.metrics as skm
    import matplotlib.pyplot as plt
    predictions = [
        [True, False, False, False, True, False, False, True, False, False, False, True, True, True, False, True, False, False, False, True],
        [0.5, 0.3, 0.7, 0.38, 0.18, 0.8, 0.3, 0.1, 0.1, 0.6, 0.1, 0.4, 0.7, 0.3, 0.1, 0.01, 0.75, 0.21, 0.5, 0.67]]
    gold = [True, True, True, False, False, False, False, True, True, True, True, False, False, True, False, False, False, True, True, True]

    evaluator = Evaluator()
    prec, rec, _ = evaluator.precision_recall(gold, predictions)
    auc = evaluator.auc(gold, predictions)
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0,1.05])
    plt.ylim([0,1.05])

    plt.show()

if __name__ == '__main__':
    main()