class ClassificationEngine:
    '''Engine for running a set of relation inference classifiers over a dataset.

    Methods:
        run(classifiers, dataset) -- Runs a list of classifiers over Dataset
        
        Parameters:
            classifiers -- Listlike of classifiers as arid.inference.classifiers.Classifier
            dataset -- Dataset to be classified as numpy.ndarray (2,3). Should contain Text and Hypothesis and X attribute, predicate, Y attribute for each
    
        Returns:
            A list of numpy.ndarrays (shape=(1,) dtype='bool'). 
            The list contains one array for each classifier. Each array contains either True/False for each entry in the dataset.
    '''
        
    def run(self, classifiers, dataset):
        '''Run a list of classifiers over dataset. See help(ClassificationEngine) for details'''
        return [cf.run(dataset) for cf in classifiers]
