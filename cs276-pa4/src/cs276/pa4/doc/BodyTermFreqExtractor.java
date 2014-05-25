package cs276.pa4.doc;


import cs276.pa4.Document;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by kavinyao on 5/9/14.
 */
public class BodyTermFreqExtractor extends TermFreqExtractor {
    @Override
    public Map<String, Integer> getFieldTermFreqs(Document d) {
        Map<String, List<Integer>> bodyHits = d.getBodyHits();
        if (bodyHits.size() == 0) {
            return Collections.emptyMap();
        }

        Map<String, Integer> bodyTF = new HashMap<>();

        for (Map.Entry<String, List<Integer>> et : bodyHits.entrySet()) {
            bodyTF.put(et.getKey(), et.getValue().size());
        }

        return bodyTF;
    }
}
