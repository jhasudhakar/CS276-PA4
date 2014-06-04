package cs276.pa4;

import weka.classifiers.functions.LibSVM;
import weka.core.SelectedTag;

/**
 * Created by kavinyao on 6/3/14.
 */
public class LinearSVMLearner extends SVMLearner {
    public LinearSVMLearner() {
    }

    public LinearSVMLearner(boolean testing) {
        super(testing);
    }

    @Override
    protected LibSVM getSVM() {
        LibSVM svm = new LibSVM();
        svm.setCost(1);
        svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        svm.setShrinking(false); // contributes a little speed-up (seem doesn't alter output value)
        return svm;
    }
}
