class ClassificationEngine:
    def runall(self, classifiers, dataset):
        return [self.run(cf, dataset) for cf in classifiers]
    
    def run(self, classifier, dataset):
        return classifier.run(dataset)
