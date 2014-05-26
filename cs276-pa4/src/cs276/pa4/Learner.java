package cs276.pa4;

import cs276.pa4.doc.DocField;
import cs276.pa4.doc.TermFreqExtractor;
import cs276.pa4.util.MapUtility;
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
        return sv.entrySet()
                .stream()
                .mapToDouble(ev -> ev.getValue() * MapUtility.getWithFallback(lv, ev.getKey(), 0.0))
                .sum();
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

    protected static double[] extractTfidfFeatures(Query q, Document doc, double score, Map<String, Double> idfs) {
        // get term frequencies
        Map<DocField, Map<String, Double>> tfs = new HashMap<>();
        for (DocField docField : DocField.values()) {
            TermFreqExtractor extractor = TermFreqExtractor.getExtractor(docField);
            Map<String, Double> tf = MapUtility.toDoubleMap(extractor.getTermFreqs(doc, q));
            tfs.put(docField, tf);

        }

        // get query frequencies
        Map<String, Double> tfQuery = new HashMap<>();
        Map<String, Integer> counts = MapUtility.count(q.getQueryWords());
        for (Map.Entry<String, Integer> et : counts.entrySet()) {
            String term = et.getKey();
            if (idfs.containsKey(term)) {
                tfQuery.put(term, 1.0 * counts.get(term) * idfs.get(term));
            } else {
                tfQuery.put(term, 0.0);
            }
        }

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
}
