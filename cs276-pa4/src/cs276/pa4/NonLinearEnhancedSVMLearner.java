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

        if (Util.getJavaVersion() == 8) {
            // for Java 8
            svm.setCost(100);
            svm.setGamma(0.001);
        } else {
            // for Java < 8
            svm.setCost(34);
            svm.setGamma(0.01005);
        }

        svm.setShrinking(false);
        return svm;
    }
}
