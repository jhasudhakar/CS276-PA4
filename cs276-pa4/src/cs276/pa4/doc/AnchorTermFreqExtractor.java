package cs276.pa4.doc;

import cs276.pa4.Document;
import cs276.pa4.util.MapUtility;

import java.util.*;

/**
 * Created by kavinyao on 5/9/14.
 */
public class AnchorTermFreqExtractor extends TermFreqExtractor {
    @Override
    public Map<String, Integer> getFieldTermFreqs(Document d) {
        Map<String, Integer> anchors = d.getAnchors();

        if (anchors.size() == 0) {
            return Collections.emptyMap();
        }

        List<Map<String, Integer>> termFreqsMaps = new ArrayList<Map<String, Integer>>();
        for (Map.Entry<String, Integer> et : anchors.entrySet()) {
            List<String> tokens = FieldProcessor.splitField(et.getKey());
            termFreqsMaps.add(MapUtility.magnify(termFreqsFromField(tokens), et.getValue()));
        }

        Map<String, Integer> termFreqCounts = new HashMap<String, Integer>();
        for (Map<String, Integer> termFreqsMap : termFreqsMaps) {
            for (Map.Entry<String, Integer> et : termFreqsMap.entrySet()) {
                MapUtility.incrementCount(et.getKey(), et.getValue(), termFreqCounts);
            }
        }

        return termFreqCounts;
    }
}
