package cs276.pa4.doc;

import cs276.pa4.Document;
import cs276.pa4.Query;
import cs276.pa4.util.MapUtility;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Extract raw term frequencies from a certain field.
 * Created by kavinyao on 5/8/14.
 */
public abstract class TermFreqExtractor {
    private static Map<DocField, TermFreqExtractor> extractors;

    static {
        // initialize known field vectors
        extractors = new HashMap<>();
        extractors.put(DocField.url, new URLTermFreqExtractor());
        extractors.put(DocField.title, new TitleTermFreqExtractor());
        extractors.put(DocField.header, new HeaderTermFreqExtractor());
        extractors.put(DocField.body, new BodyTermFreqExtractor());
        extractors.put(DocField.anchor, new AnchorTermFreqExtractor());
    }

    public static TermFreqExtractor getExtractor(DocField df) {
        return extractors.get(df);
    }

    /**
     * Compute raw term frequencies.
     * @param d
     * @return
     */
    protected abstract Map<String, Integer> getFieldTermFreqs(Document d);

    public Map<String, Integer> getTermFreqs(Document d, Query q) {
        Map<String, Integer> counts = getFieldTermFreqs(d);
        return filterByQueryTerms(counts, q);
    }


    // Helper methods

    /**
     * Count fieldWords and filter out query terms.
     * @param fieldWords
     * @return
     */
    protected Map<String, Integer> termFreqsFromField(List<String> fieldWords) {
        if (fieldWords.size() == 0) {
            return Collections.emptyMap();
        }

        return MapUtility.count(fieldWords);
    }

    protected Map<String, Integer> filterByQueryTerms(Map<String, Integer> counts, Query q) {
        Map<String, Integer> termFreqs = new HashMap<>();

        for (String qw : q.getQueryWords()) {
            int tf = MapUtility.getWithFallback(counts, qw, 0);
            termFreqs.put(qw, tf);
        }

        return termFreqs;
    }

}
