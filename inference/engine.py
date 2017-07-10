import sklearn.metrics as skm

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
    def aggregate(predictions, function):
        """Aggregates each sublist of predictions, using the specified function"""
        return np.array([function(sublist) for sublist in np.transpose(predictions)])

    def precision_recall(self, gold, prediction):
        prediction = self.aggregate(prediction, max)
        return skm.precision_recall_curve(gold, prediction)
    
    def auc(self, gold, prediction):
        precision, recall, _ = self.precision_recall(gold, prediction)
        return skm.auc(recall, precision)