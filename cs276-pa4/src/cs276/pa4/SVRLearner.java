package cs276.pa4;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * Created by kavinyao on 6/4/14.
 */
public class SVRLearner extends PointwiseLearner {
    @Override
    public Classifier train(Instances dataset) {
        LibSVM svm = new LibSVM();
        svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_NU_SVR, LibSVM.TAGS_SVMTYPE));
        svm.setCost(16);
        svm.setNu(0.25);
        svm.setShrinking(false); // contributes a little speed-up (seem doesn't alter output value)

        try {
            svm.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return svm;
    }
}
