package cs276.pa4;

import cs276.pa4.util.*;
import cs276.pa4.util.SerializationHelper;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.*;
import weka.core.converters.LibSVMSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public abstract class SVMLearner extends LinearLearner {
    private static int POS_INDEX = 0;
    private static int NEG_INDEX = 1;
    private boolean saveDataset = false;
    int counter = 1; // use counter to equally distribute instances
    private static String standardizeFile = "svm-standarize.ser";

    private Standardize standardize;

    public SVMLearner() {
        this(false);
    }

    public SVMLearner(boolean testing) {
        if (testing) {
            // load Standardize from disk
            standardize = (Standardize) SerializationHelper.loadObjectFromFile(standardizeFile);
        }
    }

    @Override
    protected ArrayList<Attribute> getAttributes() {
        ArrayList<Attribute> attributes = new ArrayList<>();

        ArrayList<String> labels = new ArrayList<>();
        labels.add("+1");
        labels.add("-1");

        attributes.add(new Attribute("label", labels));
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));

        return attributes;
    }

    @Override
    public Classifier train(Instances dataset) {
        LibSVM svm = getSVM();

        // train classifier
        try {
            svm.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (saveDataset) {
            // save dataset to file for parameter tuning with libsvm tools
            // Note: should save dataset after training LibSVM (otherwise will
            //       cause problems. Don't know why.)
            LibSVMSaver saver = new LibSVMSaver();
            saver.setClassIndex("first");
            saver.setInstances(dataset);
            try {
                saver.setFile(new File("train.libsvm"));
                saver.writeBatch();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return svm;
    }

    // also trains and saves the standardizer
    private Instances standardizeInstances(Instances dataset) {
        try {
            standardize = new Standardize();
            standardize.setInputFormat(dataset);
            dataset = Filter.useFilter(dataset, standardize);

            // save standardize to file
            if (!SerializationHelper.saveObjectToFile(standardize, standardizeFile)) {
                throw new Exception("cannot serialize standardize!");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return dataset;
    }

    private Instance standardizeInstance(Instance inst, Instances dataset) {
        // temporarily add instance to the dataset to avoid UnassignedDatasetException
        dataset.add(inst);

        // standardize the new instance
        try {
            if (standardize.input(dataset.lastInstance())) {
                inst = standardize.output();
            } else {
                throw new Exception("Cannot standardize instance!");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        // remove the temporarily added instance
        dataset.remove(dataset.numInstances() - 1);

        return inst;
    }

    protected abstract LibSVM getSVM();

    @Override
    protected TestFeatures extractFeatures(String datasetName, String dataFile, String relFile, Map<String, Double> idfs) {
        TestFeatures testFeatures = super.extractFeatures(datasetName, dataFile, relFile, idfs);
        if (relFile == null) {
            // when testing, extract document features direclty
            return testFeatures;
        }

        // when training, stndardize doc features and generate pair features
        // train standardizer
        final Instances features = standardizeInstances(testFeatures.features);

        Instances pairFeatures = new Instances(datasetName+"2", getAttributes(), 0);
        pairFeatures.setClassIndex(0);

        for (Map<String, Integer> docSet : testFeatures.indexMap.values()) {
            List<Pair<String, Instance>> docIndices = MapUtility.entryMap(docSet,
                    new UnaryFunction<Map.Entry<String, Integer>, Pair<String, Instance>>() {
                        @Override
                        public Pair<String, Instance> apply(Map.Entry<String, Integer> et) {
                            return new Pair<>(et.getKey(), features.instance(et.getValue()));
                        }
                    }
            );

            int L = docIndices.size();

            for (int i = 0; i < L; ++i) {
                for (int j = i + 1; j < L; ++j) {
                    if (i == j) {
                        continue;
                    }

                    double[] fs1 = docIndices.get(i).getSecond().toDoubleArray();
                    double[] fs2 = docIndices.get(j).getSecond().toDoubleArray();

                    if (fs1[0] == fs2[0]) {
                        // ignore pairs with the same weight
                        continue;
                    }

                    double[] diffFS = getFSDiff(fs1, fs2);
                    // make sure the class have evenly distributed examples
                    if (counter % 2 == 1 && diffFS[0] != POS_INDEX) {
                        flip(diffFS);
                    } else if (counter % 2 == 0 && diffFS[0] != NEG_INDEX) {
                        flip(diffFS);
                    }
                    counter++;

                    pairFeatures.add(new DenseInstance(1.0, diffFS));
                }
            }
        }

        TestFeatures tf = new TestFeatures();
        tf.features = pairFeatures;
        tf.indexMap = null; // don't need it

        return tf;
    }

    private void flip(double[] diffFS) {
        diffFS[0] = 1.0 - diffFS[0];
        int S = diffFS.length;
        for (int k = 1; k < S; ++k) {
            diffFS[k] = -diffFS[k];
        }
    }

    private double[] getFSDiff(double[] fs1, double[] fs2) {
        int S = fs1.length;
        double[] diffFS = new double[S];

        for (int k = 0; k < S; ++k) {
            diffFS[k] = fs1[k] - fs2[k];
        }

        diffFS[0] = diffFS[0] > 0 ? POS_INDEX : NEG_INDEX;
        return diffFS;
    }

    @Override
    protected int compareInstances(Classifier model, Instance inst1, Instance inst2, Instances allInstances) {
        double[] fs1 = standardizeInstance(inst1, allInstances).toDoubleArray();
        double[] fs2 = standardizeInstance(inst2, allInstances).toDoubleArray();

        double[] fsDiff = getFSDiff(fs1, fs2);
        Instance newInst = new DenseInstance(1.0, fsDiff);
        // add standardized instance (the API sucks...)
        allInstances.add(newInst);

        int ret = 0;
        try {
            double labelIdx = model.classifyInstance(allInstances.lastInstance());
            // index 0 -> +1 -> larger -> use -1 to rank higher
            ret = labelIdx == 0.0 ? -1 : 1;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return ret;
    }
}
