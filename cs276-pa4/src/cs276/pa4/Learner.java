package cs276.pa4;

import cs276.pa4.doc.DocField;
import cs276.pa4.doc.TermFreqExtractor;
import cs276.pa4.util.MapUtility;
import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class Learner {

    /**
     * Optimized dot product.
     * @param sv small vector
     * @param lv large vector
     * @return
     */
    private static double optimizedDotProduct(Map<String, Double> sv, Map<String, Double> lv) {
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
    private static double dotProduct(Map<String, Double> v1, Map<String, Double> v2) {
        // make sure v1 is smaller so that minimal computation is needed
        if (v1.size() > v2.size()) {
            return optimizedDotProduct(v2, v1);
        } else {
            return optimizedDotProduct(v1, v2);
        }
    }

    /* Construct training features matrix */
    public abstract Instances extract_train_features(String train_data_file, String train_rel_file, Map<String, Double> idfs) throws Exception;

    protected Instance createInstance(Query q, Document doc, double score, Map<String, Double> idfs) {
    /* Get term frequencies */
        Map<DocField, Map<String, Double>> tfs = new HashMap<>();
        for (DocField docField : DocField.values()) {
            TermFreqExtractor extractor = TermFreqExtractor.getExtractor(docField);
            Map<String, Double> tf = MapUtility.toDoubleMap(extractor.getTermFreqs(doc, q));
            tfs.put(docField, tf);

        }
                /* Get query frequencies */
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
                /* Add data */
        double[] instance = new double[6];
        instance[0] = dotProduct(tfQuery, tfs.get(DocField.url));
        instance[1] = dotProduct(tfQuery, tfs.get(DocField.title));
        instance[2] = dotProduct(tfQuery, tfs.get(DocField.body));
        instance[3] = dotProduct(tfQuery, tfs.get(DocField.header));
        instance[4] = dotProduct(tfQuery, tfs.get(DocField.anchor));
        instance[5] = score;
        return new DenseInstance(1.0, instance);
    }

    /* Train the model */
    public abstract Classifier training(Instances dataset);

    /* Construct testing features matrix */
    public abstract TestFeatures extract_test_features(String test_data_file, Map<String, Double> idfs);

    /* Test the model, return ranked queries */
    public abstract Map<String, List<String>> testing(TestFeatures tf, Classifier model);
}
