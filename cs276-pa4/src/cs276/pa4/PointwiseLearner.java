package cs276.pa4;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;

public class PointwiseLearner extends LinearLearner {
    @Override
    public Classifier train(Instances dataset) {
        Classifier lr = new LinearRegression();
        try {
            lr.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return lr;
    }
}
