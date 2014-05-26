package cs276.pa4;

import cs276.pa4.util.Pair;
import cs276.pa4.util.SerializationHelper;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LinearSVMLearner extends LinearLearner {
    private static int POS_INDEX = 0;
    private static int NEG_INDEX = 1;
    private static String standardizeFile = "linear-svm-standarize.ser";

    private Standardize standardize;

    public LinearSVMLearner() {
        this(false);
    }

    public LinearSVMLearner(boolean testing) {
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
        LibSVM svm = new LibSVM();
        // svm.setCost(C);
        // svm.setGamma(gamma); // only matter for RBF kernel
        svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
        svm.setShrinking(false);

        // filter dataset
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

        // train classifier
        try {
            svm.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return svm;
    }

    @Override
    protected int extractFeaturesFromDocs(Instances features, Query q, Map<String, Document> docs,
                                          Map<String, Double> idfs, Map<String, Map<String, Double>> rels,
                                          Map<String, Map<String, Integer>> indexMap, int index) {
        if (rels == null) {
            // for testing, extract TF-IDF features directly
            return super.extractFeaturesFromDocs(features, q, docs, idfs, rels, indexMap, index);
        }

        // for training, extract pairwise features

        String query = q.getOriginalQuery();

        List<Pair<String, double[]>> docFeatures = new ArrayList<>();
        for (Map.Entry<String, Document> entry : docs.entrySet()) {
            String url = entry.getKey();
            Document doc = entry.getValue();

            double relScore = rels.get(query).get(url);
            double[] docFS = extractTfidfFeatures(q, doc, relScore, idfs);
            docFeatures.add(new Pair<>(url, docFS));
        }

        Map<String, Integer> indices = new HashMap<>();
        int L = docFeatures.size();
        // randomly select half of all pairs
        // assume entries in docs are randomized
        for (int i = 0; i < L; ++i) {
            for (int j = i + 1; j < L; ++j) {
                if (i == j) {
                    continue;
                }

                String url1 = docFeatures.get(i).getFirst();
                String url2 = docFeatures.get(j).getFirst();
                double[] fs1 = docFeatures.get(i).getSecond();
                double[] fs2 = docFeatures.get(j).getSecond();

                if (fs1[0] == fs2[0]) {
                    // ignore pairs with the same weight
                    continue;
                }

                double[] diffFS = getFSDiff(fs1, fs2);

                features.add(new DenseInstance(1.0, diffFS));
                indices.put(url1 + "|" + url2, index++);
            }
        }

        indexMap.put(query, indices);

        return index;
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
        double[] fs1 = inst1.toDoubleArray();
        double[] fs2 = inst2.toDoubleArray();
        double[] fsDiff = getFSDiff(fs1, fs2);
        Instance newInst = new DenseInstance(1.0, fsDiff);

        // add new instance to all instances to avoid UnassignedDatasetException
        allInstances.add(newInst);

        // standardize the new instance
        try {
            if (standardize.input(allInstances.lastInstance())) {
                newInst = standardize.output();
            } else {
                throw new Exception("Cannot standardize instance!");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

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
