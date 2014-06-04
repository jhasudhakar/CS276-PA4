package cs276.pa4;

import weka.classifiers.functions.LibSVM;

/**
 * Created by kavinyao on 5/27/14.
 */
public class NonLinearSVMLearner extends SVMLearner {
    public NonLinearSVMLearner() {
    }

    public NonLinearSVMLearner(boolean testing) {
        super(testing);
    }

    protected LibSVM getSVM() {
        LibSVM svm = new LibSVM();
        svm.setCost(34.0);
        svm.setGamma(0.01);
        svm.setShrinking(false);
        return svm;
    }
}
