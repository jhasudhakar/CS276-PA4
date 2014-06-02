package cs276.pa4;

import cs276.pa4.util.ListUtility;
import cs276.pa4.util.MapUtility;
import cs276.pa4.util.Pair;
import cs276.pa4.util.UnaryFunction;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

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
                                         final Classifier model, final Instances allFeatures) {
        List<Pair<String, Integer>> pairs = MapUtility.toPairs(instances);

        Collections.sort(pairs, new Comparator<Pair<String, Integer>>() {
            @Override
            public int compare(Pair<String, Integer> p1, Pair<String, Integer> p2) {
                Instance inst1 = allFeatures.instance(p1.getSecond());
                Instance inst2 = allFeatures.instance(p2.getSecond());

                return compareInstances(model, inst1, inst2, allFeatures);
            }
        });

        return ListUtility.map(pairs, new UnaryFunction<Pair<String, Integer>, String>() {
            @Override
            public String apply(Pair<String, Integer> p) {
                return p.getFirst();
            }
        });
    }

    protected abstract int compareInstances(Classifier model, Instance inst1, Instance inst2, Instances allInstances);
}
