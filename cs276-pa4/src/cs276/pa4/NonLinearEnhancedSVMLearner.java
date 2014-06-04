package cs276.pa4;

import weka.classifiers.functions.LibSVM;

/**
 * Created by kavinyao on 6/3/14.
 */
public class NonLinearEnhancedSVMLearner extends EnhancedSVMLearner {
    public NonLinearEnhancedSVMLearner() {
    }

    public NonLinearEnhancedSVMLearner(boolean testing) {
        super(testing);
    }

    @Override
    protected LibSVM getSVM() {
        LibSVM svm = new LibSVM();
        svm.setCost(100);
        svm.setGamma(0.001);
        svm.setShrinking(false);
        return svm;
    }
}
