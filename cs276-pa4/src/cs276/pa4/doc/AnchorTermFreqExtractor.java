package cs276.pa4.doc;

import cs276.pa4.Document;
import cs276.pa4.util.MapUtility;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

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

        return anchors.entrySet()
                .stream()
                .map(et -> {
                    List<String> tokens = FieldProcessor.splitField(et.getKey());
                    return MapUtility.magnify(termFreqsFromField(tokens), et.getValue());
                })
                .flatMap(m -> m.entrySet().stream())
                .collect(Collectors.groupingBy(Map.Entry::getKey,
                         Collectors.summingInt(Map.Entry::getValue)));
    }
}
