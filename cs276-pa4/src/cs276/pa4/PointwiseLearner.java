package cs276.pa4;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
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

    @Override
    protected int compareInstances(Classifier model, Instance inst1, Instance inst2, Instances allInstances) {
        Double score1 = getScore(model, inst1);
        Double score2 = getScore(model, inst2);

        // rank larger values higher
        return score2.compareTo(score1);
    }

    protected static double getScore(Classifier model, Instance inst) {
        double score = 0;
        try {
            score = model.classifyInstance(inst);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return score;
    }
}
