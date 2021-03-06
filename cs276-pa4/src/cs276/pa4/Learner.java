package cs276.pa4;

import cs276.pa4.doc.DocField;
import cs276.pa4.doc.TermFreqExtractor;
import cs276.pa4.util.MapUtility;
import cs276.pa4.util.UnaryFunction;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static cs276.pa4.Util.loadRelData;
import static cs276.pa4.Util.loadTrainData;

public abstract class Learner {
    private static double SMOOTH_BODY_LENGTH = 900;

    /* Construct training features matrix */
    public Instances extractTrainFeatures(String trainDataFile, String trainRelFile, Map<String, Double> idfs) {
        TestFeatures testFeatures = extractFeatures("train-dataset", trainDataFile, trainRelFile, idfs);
        return testFeatures.features;
    }

    /* Train the model */
    public abstract Classifier train(Instances dataset);

    /* Construct testing features matrix */
    public TestFeatures extractTestFeatures(String testDataFile, Map<String, Double> idfs) {
        return extractFeatures("test-dataset", testDataFile, null, idfs);
    }

    // always extract indices
    // let the caller decide whether to use them
    protected TestFeatures extractFeatures(String datasetName, String dataFile, String relFile, Map<String, Double> idfs) {
        ArrayList<Attribute> attributes = getAttributes();
        Instances features = new Instances(datasetName, attributes, 0);

        Map<Query, Map<String, Document>> data = loadTrainData(dataFile);
        Map<String, Map<String, Double>> rels = null;
        if (relFile != null) {
            rels = loadRelData(relFile);
        }

        processDocuments(data);

        Map<String, Map<String, Integer>> indexMap = new HashMap<>();
        int index = 0;
        for (Map.Entry<Query, Map<String, Document>> entry : data.entrySet()) {
            Query q = entry.getKey();
            Map<String, Document> docs = entry.getValue();
            index = extractFeaturesFromDocs(features, q, docs, idfs, rels, indexMap, index);
        }

        features.setClassIndex(0);

        TestFeatures testFeatures = new TestFeatures();
        testFeatures.features = features;
        testFeatures.indexMap = indexMap;

        return testFeatures;
    }

    /**
     * Hook after documents are laoded.
     * Overridable by subclass.
     * @param data
     */
    protected void processDocuments(Map<Query, Map<String, Document>> data) {
        // do nothing
    }

    /* Test the model, return ranked queries */
    public Map<String, List<String>> test(TestFeatures tf, Classifier model) {
        Map<String, List<String>> rankedResults = new HashMap<>();
        Instances allFeatures = tf.features;

        for (Map.Entry<String, Map<String, Integer>> entry : tf.indexMap.entrySet()) {
            String query = entry.getKey();
            Map<String, Integer> instances = entry.getValue();
            List<String> rankings = rankDocuments(query, instances, model, allFeatures);
            rankedResults.put(query, rankings);
        }

        return rankedResults;
    }

    /**
     * @param features
     * @param q
     * @param docs
     * @param idfs
     * @param rels if null, extracting for test set
     * @param indexMap
     * @param index   @return next available index
     */
    protected abstract int extractFeaturesFromDocs(Instances features, Query q, Map<String, Document> docs, Map<String, Double> idfs, Map<String, Map<String, Double>> rels, Map<String, Map<String, Integer>> indexMap, int index);

    /**
     * The class index must be 0.
     * @return
     */
    protected abstract ArrayList<Attribute> getAttributes();

    protected abstract List<String> rankDocuments(String query, Map<String, Integer> instances, Classifier model, Instances allFeatures);

    // helper methods

    /**
     * Optimized dot product.
     * @param sv small vector
     * @param lv large vector
     * @return
     */
    protected static double optimizedDotProduct(Map<String, Double> sv, Map<String, Double> lv) {
        double sum = 0.0;
        for (Map.Entry<String, Double> et : sv.entrySet()) {
            sum += et.getValue() * MapUtility.getWithFallback(lv, et.getKey(), 0.0);
        }

        return sum;
    }

    /**
     * Compute dot product of two sparse vectors.
     * @param v1 sparse vector
     * @param v2 sparse vector
     * @return
     */
    protected static double dotProduct(Map<String, Double> v1, Map<String, Double> v2) {
        // make sure v1 is smaller so that minimal computation is needed
        if (v1.size() > v2.size()) {
            return optimizedDotProduct(v2, v1);
        } else {
            return optimizedDotProduct(v1, v2);
        }
    }

    private static Map<String, Double> sublinear(Map<String, Double> vals) {
        return MapUtility.iMap(vals, new UnaryFunction<Double, Double>() {
            @Override
            public Double apply(Double val) {
                return val == 0.0 ? 0.0 : 1 + Math.log(val);
            }
        });
    }

    private static Map<String, Double> normalizeTF(Map<String, Double> tf, Document d) {
//        return sublinear(tf);
//        return lengthNormalize(tf, d);
        return lengthNormalize(sublinear(tf), d);
//        return tf;
    }

    /**
     * Normalize by body length.
     * @param termFreqs
     * @param d
     * @return
     */
    private static Map<String, Double> lengthNormalize(Map<String, Double> termFreqs, Document d) {
        final double smoothedBodyLength = d.getBodyLength() + SMOOTH_BODY_LENGTH;
        return MapUtility.iMap(termFreqs, new UnaryFunction<Double, Double>() {
            @Override
            public Double apply(Double f) {
                return f / smoothedBodyLength;
            }
        });
    }

    protected static double[] extractTfidfFeatures(Query q, Document doc, double score, Map<String, Double> idfs) {
        Map<DocField, Map<String, Double>> tfs = getRawDocTermFreqs(q, doc);
        Map<String, Double> tfQuery = getQueryFreqs(q, idfs);

        // add data
        double[] instance = new double[6];
        instance[0] = score;
        instance[1] = dotProduct(tfQuery, tfs.get(DocField.url));
        instance[2] = dotProduct(tfQuery, tfs.get(DocField.title));
        instance[3] = dotProduct(tfQuery, tfs.get(DocField.body));
        instance[4] = dotProduct(tfQuery, tfs.get(DocField.header));
        instance[5] = dotProduct(tfQuery, tfs.get(DocField.anchor));

        return instance;
    }

    protected static Map<DocField, Map<String, Double>> getRawDocTermFreqs(Query q, Document doc) {
        // get term frequencies
        Map<DocField, Map<String, Double>> tfs = new HashMap<>();
        for (DocField docField : DocField.values()) {
            TermFreqExtractor extractor = TermFreqExtractor.getExtractor(docField);
            Map<String, Double> tf = normalizeTF(MapUtility.toDoubleMap(extractor.getTermFreqs(doc, q)), doc);
            tfs.put(docField, tf);
        }
        return tfs;
    }

    protected static Map<String, Double> getQueryFreqs(Query q, Map<String, Double> idfs) {
        // get query frequencies
        Map<String, Double> tfQuery = new HashMap<>();
        Map<String, Integer> counts = MapUtility.count(q.getQueryWords());

        for (Map.Entry<String, Integer> et : counts.entrySet()) {
            String term = et.getKey();
            if (idfs.containsKey(term)) {
                tfQuery.put(term, 1.0 * counts.get(term) * idfs.get(term));
            } else {
                tfQuery.put(term, Math.log(Util.NDocs));
            }
        }

        return tfQuery;
    }
}
