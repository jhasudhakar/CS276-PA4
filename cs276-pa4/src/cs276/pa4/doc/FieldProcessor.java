package cs276.pa4.doc;

import java.util.*;

/**
 * Created by kavinyao on 5/10/14.
 */
public class FieldProcessor {
    protected static final String NON_WORD_REGEX = "\\W+";
    protected static final String WHITE_SPACE_REGEX = "\\s+";

    /**
     * URL most not be null.
     * @param url
     * @return
     */
    public static List<String> splitURL(String url) {
        return tokenize(url, NON_WORD_REGEX);
    }

    /**
     * Generic method for splitting field.
     * @param field
     * @return
     */
    public static List<String> splitField(String field) {
        if (field == null) {
            return Collections.emptyList();
        }

        return tokenize(field, WHITE_SPACE_REGEX);
    }

    /**
     * title must be in lowercased if not null.
     * @param title
     * @return
     */
    public static List<String> splitTitle(String title) {
        return splitField(title);
    }

    public static List<String> splitHeaders(List<String> headers) {
        if (headers.size() == 0) {
            return Collections.emptyList();
        }

        List<String> headerWords = new ArrayList<>();

        for (String header : headers) {
            headerWords.addAll(splitField(header));
        }

        return headerWords;
    }

    public static List<String> splitAnchors(Map<String, Integer> anchors) {
        if (anchors.size() == 0) {
            return Collections.emptyList();
        }

        List<String> anchorWords = new ArrayList<>();

        for (String anchorText : anchors.keySet()) {
            anchorWords.addAll(splitField(anchorText));
        }

        return anchorWords;
    }

    protected static List<String> tokenize(String s, String sep) {
        return Arrays.asList(s.split(sep));
    }
}
