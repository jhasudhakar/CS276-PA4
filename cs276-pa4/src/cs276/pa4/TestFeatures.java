package cs276.pa4;

import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;

public class TestFeatures {

    public TestFeatures() {
        this.features = null;
        this.indexMap = new HashMap<>();
    }

    /* Test features */
    Instances features;

    /* Associate query-doc pair to its index within FEATURES instances
     * {query -> {doc -> index}}
     *
     * For example, you can get the feature for a pair of (query, url) using:
     *   features.get(indexMap.get(query).get(url));
     * */
    Map<String, Map<String, Integer>> indexMap;
}
