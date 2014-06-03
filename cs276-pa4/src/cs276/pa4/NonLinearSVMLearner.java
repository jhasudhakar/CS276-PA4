package cs276.pa4;

import weka.classifiers.functions.LibSVM;

/**
 * Created by kavinyao on 5/27/14.
 */
public class NonLinearSVMLearner extends LinearSVMLearner {
    public NonLinearSVMLearner() {
    }

    public NonLinearSVMLearner(boolean testing) {
        super(testing);
    }

    protected LibSVM getSVM() {
        LibSVM svm = new LibSVM();
        svm.setCost(8.0);
        svm.setGamma(0.001);
        svm.setShrinking(false);
        return svm;
    }
}
