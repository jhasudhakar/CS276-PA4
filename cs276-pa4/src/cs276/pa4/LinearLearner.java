package cs276.pa4;

import cs276.pa4.doc.DocField;
import cs276.pa4.doc.TermFreqExtractor;
import cs276.pa4.util.MapUtility;
import cs276.pa4.util.Pair;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by kavinyao on 5/25/14.
 */
public abstract class LinearLearner extends Learner {
    @Override
    protected int extractFeaturesFromDocs(Instances features, Query q, Map<String, Document> docs,
                                          Map<String, Double> idfs, Map<String, Map<String, Double>> rels,
                                          Map<String, Map<String, Integer>> indexMap, int index) {
        Map<String, Integer> indices = new HashMap<>();
        String query = q.getOriginalQuery();

        for (Map.Entry<String, Document> entry : docs.entrySet()) {
            String url = entry.getKey();
            Document doc = entry.getValue();

            double relScore = rels == null ? 0.0 : rels.get(query).get(url);
            Instance inst = new DenseInstance(1.0, extractTfidfFeatures(q, doc, relScore, idfs));
            features.add(inst);
            indices.put(url, index++);
        }

        indexMap.put(query, indices);

        return index;
    }

    @Override
    protected ArrayList<Attribute> getAttributes() {
        ArrayList<Attribute> attributes = new ArrayList<>();

        attributes.add(new Attribute("relevance_score"));
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));

        return attributes;
    }

    @Override
    protected List<String> rankDocuments(String query, Map<String, Integer> instances,
                                         Classifier model, Instances allFeatures) {
        List<Pair<String, Double>> scores = new ArrayList<>();

            /* For each query document pair, estimate relevance score */
        for (Map.Entry<String, Integer> et : instances.entrySet()) {
            String url = et.getKey();
            int index = et.getValue();

            double relScore = 0;
            try {
                relScore = model.classifyInstance(allFeatures.instance(index));
            } catch (Exception e) {
                e.printStackTrace();
            }
            scores.add(new Pair<>(url, relScore));
        }

            /* Sort documents */
        Collections.sort(scores, (p1, p2) -> {
            Double s1 = p1.getSecond();
            Double s2 = p2.getSecond();
            return -s1.compareTo(s2);
        });

        return scores
                .stream()
                .map(Pair<String, Double>::getFirst)
                .collect(Collectors.toList());
    }

    protected double[] extractTfidfFeatures(Query q, Document doc, double score, Map<String, Double> idfs) {
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
