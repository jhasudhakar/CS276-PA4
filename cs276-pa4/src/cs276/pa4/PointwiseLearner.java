package cs276.pa4;

import cs276.pa4.doc.DocField;
import cs276.pa4.util.MapUtility;
import cs276.pa4.util.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

import cs276.pa4.doc.TermFreqExtractor;

import static cs276.pa4.Util.loadRelData;
import static cs276.pa4.Util.loadTrainData;

public class PointwiseLearner extends Learner {

    @Override
    public Instances extract_train_features(String train_data_file,
                                            String train_rel_file, Map<String, Double> idfs) throws Exception {
        Instances dataset = null;
		
		/* Build attributes list */
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));
        attributes.add(new Attribute("relevance_score"));
        dataset = new Instances("train_dataset", attributes, 0);

        /* Load train data. */
        Map<Query, Map<String, Document>> trainData;
        Map<String, Map<String, Double>> relData;
        trainData = loadTrainData(train_data_file);
        relData = loadRelData(train_rel_file);

        for (Map.Entry<Query, Map<String, Document>> entry : trainData.entrySet()) {
            Query q = entry.getKey();
            Map<String, Document> docs = entry.getValue();
            for (String url : docs.keySet()) {
                Document doc = docs.get(url);
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
                instance[5] = relData.get(q.getOriginalQuery()).get(url);
                Instance inst = new DenseInstance(1.0, instance);
                dataset.add(inst);
            }
        }

		/* Set last attribute as target */
        dataset.setClassIndex(dataset.numAttributes() - 1);

        return dataset;
    }

    @Override
    public Classifier training(Instances dataset) {
        Classifier lr = new LinearRegression();
        try {
            lr.buildClassifier(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return lr;
    }

    @Override
    public TestFeatures extract_test_features(String test_data_file,
                                              Map<String, Double> idfs) {
        int index = 0;
        TestFeatures testFeatures = new TestFeatures();

		/* Build attributes list */
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));
        attributes.add(new Attribute("relevance_score"));
        testFeatures.features = new Instances("test_dataset", attributes, 0);

        Map<Query, Map<String, Document>> testData = null;
        try {
            testData = loadTrainData(test_data_file);
        } catch (Exception e) {
            e.printStackTrace();
        }
        for (Map.Entry<Query, Map<String, Document>> entry : testData.entrySet()) {
            Query q = entry.getKey();
            testFeatures.index_map.put(q.getOriginalQuery(), new HashMap<>());
            Map<String, Document> docs = entry.getValue();
            for (String url : docs.keySet()) {
                Document doc = docs.get(url);
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
                instance[5] = 0;
                Instance inst = new DenseInstance(1.0, instance);
                testFeatures.features.add(inst);
                testFeatures.index_map.get(q.getOriginalQuery()).put(url, index);
                index++;
            }
        }

        testFeatures.features.setClassIndex(testFeatures.features.numAttributes() - 1);

        return testFeatures;
    }

    @Override
    public Map<String, List<String>> testing(TestFeatures tf,
                                             Classifier model) {
        Map<String, List<String>> rankedQueries = new HashMap<>();

        for (Map.Entry<String, Map<String, Integer>> entry : tf.index_map.entrySet()) {
            String query = entry.getKey();
            Map<String, Integer> urlIndex = entry.getValue();
            List<Pair<String, Double>> scores = new ArrayList<>();

            /* For each query document pair, estimate relevance score */
            for (Map.Entry<String, Integer> et : urlIndex.entrySet()) {
                String url = et.getKey();
                int index = et.getValue();
                double relScore = 0;
                try {
                    relScore = model.classifyInstance(tf.features.instance(index));
                } catch (Exception e) {
                    e.printStackTrace();
                }
                scores.add(new Pair<>(url, relScore));
            }

            /* Sort documents */
            Collections.sort(scores, (p1, p2) -> {
                double s1 = p1.getSecond();
                double s2 = p2.getSecond();
                if (s1 < s2) {
                    return 1;
                } else if (s1 > s2) {
                    return -1;
                } else {
                    return 0;
                }
            });

            /* Put it into rankedQueries */
            List<String> rankings = scores.stream().map(Pair<String, Double>::getFirst).collect(Collectors.toList());
            rankedQueries.put(query, rankings);
        }

        return rankedQueries;
    }

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
}
