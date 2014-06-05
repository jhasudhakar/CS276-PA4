package cs276.pa4;

import weka.classifiers.functions.LibSVM;

/**
 * Created by kavinyao on 6/4/14.
 */
public class CornNonLinearEnhancedSVMLearner extends EnhancedSVMLearner {
    public CornNonLinearEnhancedSVMLearner() {
    }

    public CornNonLinearEnhancedSVMLearner(boolean testing) {
        super(testing);
    }

    @Override
    protected LibSVM getSVM() {
        LibSVM svm = new LibSVM();
        svm.setCost(34);
        svm.setGamma(0.01005);
        svm.setShrinking(false);
        return svm;
    }
}
