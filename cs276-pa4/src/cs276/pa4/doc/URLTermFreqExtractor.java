package cs276.pa4.doc;


import cs276.pa4.Document;

import java.util.List;
import java.util.Map;

/**
 * Created by kavinyao on 5/8/14.
 */
public class URLTermFreqExtractor extends TermFreqExtractor {
    @Override
    public Map<String, Integer> getFieldTermFreqs(Document d) {
        List<String> urlWords = d.getFieldTokens(DocField.url);
        return termFreqsFromField(urlWords);
    }
}
