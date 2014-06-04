package cs276.pa4;

import cs276.pa4.doc.DocField;
import cs276.pa4.doc.FieldProcessor;
import cs276.pa4.util.*;

import java.util.*;

public class Document {
    // fields
    // all fields are trimmed and in lowercase
    private String originalURL = null;
    private String url = null;
    private String title = null;
    private List<String> headers = new ArrayList<>();
    // term -> [list of positions]
    private Map<String, List<Integer>> bodyHits = new HashMap<>();
    // anchor text -> reference count
    private Map<String, Integer> anchors = new HashMap<>();

    // other attributes
    private int bodyLength = 0;
    private int pageRank = 0;

    // cached tokens: field -> [tokens]
    private Map<DocField, List<String>> fieldTokens;

    // for smallest window
    private List<List<Pair<Integer, String>>> possibleWindows;
    private List<Pair<Integer, String>> bodyTermPositions;

    public Document(String url) {
        this.originalURL = url;
        this.url = normalize(url);
        this.fieldTokens = new HashMap<>();
        this.possibleWindows = new ArrayList<>();
    }

    private static String normalize(String s) {
        return s.trim().toLowerCase();
    }

    public Map<String, Integer> getAnchors() {
        return anchors;
    }

    public Map<String, List<Integer>> getBodyHits() {
        return bodyHits;
    }

    public void setTitle(String title) {
        this.title = normalize(title);
    }

    public int getBodyLength() {
        return bodyLength;
    }

    public String getOriginalURL() {
        return originalURL;
    }

    public void addHeader(String header) {
        this.headers.add(normalize(header));
    }

    public void addBodyHits(String term, List<Integer> positions) {
        term = normalize(term);

        if (this.bodyHits.containsKey(term)) {
            throw new IllegalStateException("Duplicate term in body");
        }

        this.bodyHits.put(term, positions);
    }

    public void setBodyLength(int bodyLength) {
        this.bodyLength = bodyLength;
    }

    public int getPageRank() {
        return pageRank;
    }

    public void setPageRank(int pageRank) {
        this.pageRank = pageRank;
    }

    public void addAnchor(String anchorText, int count) {
        anchorText = normalize(anchorText);

        if (this.anchors.containsKey(anchorText)) {
            throw new IllegalStateException("Duplicate anchor text");
        }

        this.anchors.put(anchorText, count);
    }

    /**
     * Call this method after this document is constructed.
     */
    public void end() {
        // cache tokens of each field
        fieldTokens.put(DocField.url, FieldProcessor.splitURL(this.url));
        fieldTokens.put(DocField.title, FieldProcessor.splitTitle(this.title));
        fieldTokens.put(DocField.header, FieldProcessor.splitHeaders(this.headers));

        // cache unique terms and position of each term in each field
        // cache the values to speed up future getWindow calls
        possibleWindows.add(translateList(fieldTokens.get(DocField.url)));
        possibleWindows.add(translateList(fieldTokens.get(DocField.title)));

        for (String header : headers) {
            possibleWindows.add(translateList(FieldProcessor.splitField(header)));
        }

        for (String anchorText : anchors.keySet()) {
            possibleWindows.add(translateList(FieldProcessor.splitField(anchorText)));
        }

        bodyTermPositions = new ArrayList<Pair<Integer, String>>();
        // generate <position, term> pairs
        for (Map.Entry<String, List<Integer>> et : bodyHits.entrySet()) {
            final String term = et.getKey();
            bodyTermPositions.addAll(ListUtility.map(et.getValue(),
                    new UnaryFunction<Integer, Pair<Integer, String>>() {
                        @Override
                        public Pair<Integer, String> apply(Integer i) {
                            return new Pair<Integer, String>(i, term);
                        }
                    }
            ));
        }

        // sort them
        Collections.sort(bodyTermPositions, new Comparator<Pair<Integer, String>>() {
            @Override
            public int compare(Pair<Integer, String> p1, Pair<Integer, String> p2) {
                return p1.getFirst().compareTo(p2.getFirst());
            }
        });
    }

    /**
     * Translate a List of String to a list of <index, term> pairs
     * @param terms
     * @return
     */
    private List<Pair<Integer, String>> translateList(List<String> terms) {
        List<Pair<Integer, String>> positions = new ArrayList<Pair<Integer, String>>();
        for (int i = 0; i < terms.size(); ++i) {
            positions.add(new Pair<Integer, String>(i, terms.get(i)));
        }

        return positions;
    }


    /**
     * A generic minimal window computation algorithm.
     * Ref: http://stackoverflow.com/a/3592255/1240620
     * @param positions positions of obejcts in sequence
     * @param objects the objects to be included in the window
     * @param <T>
     * @return the minimal window
     */
    private static <T> int getWindow(List<Pair<Integer, T>> positions, Set<T> objects) {
        Set<T> uniques = new HashSet<T>(ListUtility.map(positions,
                new UnaryFunction<Pair<Integer, T>, T>() {
                    @Override
                    public T apply(Pair<Integer, T> p) {
                        return p.getSecond();
                    }
                }
        ));

        if (!uniques.containsAll(objects)) {
            return -1;
        }

        int window = positions.size();
        Map<T, Integer> indices = new LinkedHashMap<>();

        for (Pair<Integer, T> position : positions) {
            int j = position.getFirst();
            T obj = position.getSecond();

            if (objects.contains(obj)) {
                indices.remove(obj);
                indices.put(obj, j);
                if (indices.size() == objects.size()) {
                    int i = indices.values().iterator().next();
                    if (window > j - i + 1) {
                        window = j - i + 1;
                    }
                }
            }
        }

        return window;
    }

    /**
     * Get smallest window in body.
     * @param terms
     * @return
     */
    private int getBodyWindow(Set<String> terms) {
        return getWindow(bodyTermPositions, terms);
    }

    /**
     * Compute the smallest window containing all the terms.
     * smallest-window(d, terms) = min(window(f, terms)) for f of d.fields
     * @param termSet unique terms
     * @return the smallest window, or -1 if no smallest window found
     */
    public int getSmallestWindow(final Set<String> termSet) {
        List<Integer> smallestWindows = ListUtility.filter(
                ListUtility.map(possibleWindows,
                        new UnaryFunction<List<Pair<Integer, String>>, Integer>() {
                            @Override
                            public Integer apply(List<Pair<Integer, String>> positions) {
                                return getWindow(positions, termSet);
                            }
                        }
                ),
                new Predicate<Integer>() {
                    @Override
                    public boolean test(Integer i) {
                        return i > 0;
                    }
                }
        );

        int sw = -1;

        if (smallestWindows.size() > 0) {
            sw = Collections.min(smallestWindows);
        }

        int bodySw = getBodyWindow(termSet);

        if (sw == -1 || sw > bodySw) {
            sw = bodySw;
        }

        return sw;
    }

    public List<String> getFieldTokens(DocField f) {
        if (f == DocField.body || f == DocField.anchor) {
            throw new IllegalStateException("Cannot get tokens of body.");
        }

        return fieldTokens.get(f);
    }

    public int getNumFieldTokens(DocField f) {
        if (f == DocField.body) {
            return this.bodyLength;
        } else if (f == DocField.anchor) {
            List<Integer> counts = new ArrayList<Integer>();
            for (Map.Entry<String, Integer> et : anchors.entrySet()) {
                List<String> tokens = FieldProcessor.splitField(et.getKey());
                counts.addAll(MapUtility.magnify(MapUtility.count(tokens), et.getValue()).values());
            }

            return ListUtility.sum(counts);
        }

        return fieldTokens.get(f).size();
    }

    // For debug
    public String toString() {
        StringBuilder result = new StringBuilder();

        String NEW_LINE = System.getProperty("line.separator");
        if (title != null) result.append("title: " + title + NEW_LINE);
        if (headers.size() > 0) result.append("headers: " + headers.toString() + NEW_LINE);
        if (bodyHits.size() > 0) result.append("body_hits: " + bodyHits.toString() + NEW_LINE);
        if (bodyLength != 0) result.append("body_length: " + bodyLength + NEW_LINE);
        if (pageRank != 0) result.append("page_rank: " + pageRank + NEW_LINE);
        if (anchors.size() > 0) result.append("anchors: " + anchors.toString() + NEW_LINE);

        return result.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }

        if (obj == null || !(obj instanceof Document)) {
            return false;
        }

        Document another = (Document) obj;
        return this.url.equals(another.url);
    }

    @Override
    public int hashCode() {
        return url.hashCode();
    }
}
