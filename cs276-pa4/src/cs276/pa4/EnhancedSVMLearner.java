package cs276.pa4;

import cs276.pa4.doc.DocField;
import cs276.pa4.util.MapUtility;
import weka.core.Attribute;

import java.util.*;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

/**
 * Created by kavinyao on 6/3/14.
 */
public abstract class EnhancedSVMLearner extends SVMLearner {
    private static Map<DocField, Double> Bf;
    private static Map<DocField, Double> Wf;
    private double K1 = 2.5;
    private double lambda = 1.6;
    private double lambdaPrime = 2.1;

    // initialize weights
    static {
        Bf = new HashMap<>();
        Bf.put(DocField.url, 1.0);
        Bf.put(DocField.title, 0.9);
        Bf.put(DocField.header, 0.5);
        Bf.put(DocField.body, 1.0);
        Bf.put(DocField.anchor, 0.2);

        Wf = new HashMap<>();
        Wf.put(DocField.url, 3.0);
        Wf.put(DocField.title, 3.1);
        Wf.put(DocField.header, 1.6);
        Wf.put(DocField.body, 0.1);
        Wf.put(DocField.anchor, 1.6);
    }

    // instance variables
    // field -> document -> length
    Map<DocField, Map<Document, Double>> lengths;
    // field -> avg. length
    Map<DocField, Double> avgLengths;
    // document -> pagerank
    Map<Document, Double> pagerankScores;

    public EnhancedSVMLearner() {
    }

    public EnhancedSVMLearner(boolean testing) {
        super(testing);
    }

    @Override
    protected ArrayList<Attribute> getAttributes() {
        ArrayList<Attribute> attributes = new ArrayList<>();

        ArrayList<String> labels = new ArrayList<>();
        labels.add("+1");
        labels.add("-1");
        attributes.add(new Attribute("label", labels));

        ArrayList<String> binary = new ArrayList<>();
        binary.add("0");
        binary.add("1");

        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));

        attributes.add(new Attribute("b_url_w"));
        attributes.add(new Attribute("b_title_w"));
        attributes.add(new Attribute("b_body_w"));
        attributes.add(new Attribute("b_header_w"));
        attributes.add(new Attribute("b_anchor_w"));

        attributes.add(new Attribute("bm25_w"));
        attributes.add(new Attribute("window_w"));
        attributes.add(new Attribute("pagerank_w"));

        attributes.add(new Attribute("url_len_w"));
        attributes.add(new Attribute("log_body_len_w"));
        attributes.add(new Attribute("is_pdf", binary));
        attributes.add(new Attribute("with_cgi-bin", binary));
        attributes.add(new Attribute("got_all_query_words", binary));

        return attributes;
    }

    /**
     * Calculate average lengths of document fields.
     * @param queryDict
     */
    @Override
    protected void processDocuments(Map<Query, Map<String, Document>> queryDict) {
        List<Document> uniqueDocs = queryDict.values()
                .stream()
                .flatMap(m -> m.values().stream())
                .distinct()
                .collect(toList());

        // compute length of each field
        lengths = new HashMap<>();
        for (DocField f : DocField.values()) {
            lengths.put(f, lengthsOfField(uniqueDocs, d -> d.getNumFieldTokens(f)));
        }

        // compute average lengths of each field
        avgLengths = new HashMap<>();
        for (DocField docField : DocField.values()) {
            avgLengths.put(docField, averageFieldLength(lengths.get(docField)));
            //System.err.println("avg(" + docField + ") = " + avgLengths.get(docField));
        }

        pagerankScores = uniqueDocs
                .stream()
                .collect(toMap(Function.identity(), d -> new Double(d.getPageRank())));
    }

    /**
     * Compute length of given field for every document.
     * @param docs must not contain duplicate
     * @param getLength length getter for field
     * @return
     */
    private static Map<Document, Double> lengthsOfField(List<Document> docs,
                                                        Function<Document, Integer> getLength) {
        return docs
                .stream()
                .collect(toMap(Function.identity(), d -> getLength.apply(d).doubleValue()));
    }

    private static Double averageFieldLength(Map<Document, Double> fieldLengths) {
        return fieldLengths.values()
                .stream()
                        // cannot use Function.identity as a ToDoubleFunction, :(
                .mapToDouble(d -> d)
                .average()
                .getAsDouble();
    }

    private double getTermWeight(Document d, Map<DocField, Map<String, Double>> tfs, String t, Query q) {
        return Arrays.asList(DocField.values())
                .stream()
                .map(f -> {
                    double tf = MapUtility.getWithFallback(tfs.get(f), t, 0.0);
                    double denominator = 1 + Bf.get(f) * (lengths.get(f).get(d) / avgLengths.get(f) - 1);
                    double ftf = denominator == 0.0 ? 0.0 : tf / denominator;
                    return Wf.get(f) * ftf;
                })
                .mapToDouble(x -> x)
                .sum();
    }

    @Override
    protected void flip(double[] diffFS) {
        super.flip(diffFS);

        for (int k = 15; k < 19; ++k) {
            diffFS[k] = diffFS[k] == 0.0 ? 0.0 : 1.0;
        }
    }

    @Override
    protected double[] getFSDiff(double[] fs1, double[] fs2) {
        double[] diffFS = super.getFSDiff(fs1, fs2);

        for (int k = 15; k < 19; ++k) {
            diffFS[k] = diffFS[k] == 0.0 ? 0.0 : 1.0;
        }

        return diffFS;
    }

    @Override
    protected double[] extractFeaturesFromDocument(Query q, Document doc,
                                                   double score, Map<String, Double> idfs) {
        Map<DocField, Map<String, Double>> tfs = getRawDocTermFreqs(q, doc);
        Map<String, Double> tfQuery = getQueryFreqs(q, idfs);

        List<Double> features = new ArrayList<>();

        // tf-idf features
        features.add(score);

        features.add(dotProduct(tfQuery, tfs.get(DocField.url)));
        features.add(dotProduct(tfQuery, tfs.get(DocField.title)));
        features.add(dotProduct(tfQuery, tfs.get(DocField.body)));
        features.add(dotProduct(tfQuery, tfs.get(DocField.header)));
        features.add(dotProduct(tfQuery, tfs.get(DocField.anchor)));

        // add binary features
        for (int i = 1; i < 6; ++i) {
            features.add(features.get(i) > 0 ? 1.0 : 0.0);
        }

        // extended features
        features.add(getSimScore(doc, q, idfs));
        HashSet<String> termSet = new HashSet<>(q.getQueryWords());
        int window = doc.getSmallestWindow(termSet);
        features.add((double) window);
        features.add((double) doc.getPageRank());

        features.add((double) doc.getOriginalURL().length());
        features.add(Math.log(1 + doc.getBodyLength()));
        features.add(doc.getOriginalURL().endsWith(".pdf") ? 1.0 : 0.0);
        features.add(doc.getOriginalURL().contains("cgi-bin") ? 1.0 : 0.0);
        features.add(window > -1 ? 1.0 : 0.0);

        return listToArray(features);
    }

    private static double[] listToArray(List<Double> l) {
        double[] arr = new double[l.size()];
        for (int i = 0; i < l.size(); ++i) {
            arr[i] = l.get(i);
        }
        return arr;
    }

    private double V(int pageRank) {
        if (lambdaPrime < 1.0) {
            throw new IllegalStateException("lambdaPrime should be >= 1");
        }
        return Math.log(pageRank + lambdaPrime);
    }

    private double getSimScore(Document d, Query q, Map<String, Double> idfs) {
        Map<DocField, Map<String, Double>> tfs = getRawDocTermFreqs(q, d);
        Map<String, Double> tfQuery = getQueryFreqs(q, idfs);

        double bm25 = tfQuery.keySet()
                .stream()
                .map(t -> {
                    double idf = 0.0;
                    if (idfs.containsKey(t)) {
                        idf = idfs.get(t);
                    } else {
                        idf = Math.log(Util.NDocs);
                    }
                    double wdt = getTermWeight(d, tfs, t, q);
                    if (wdt + K1 == 0.0) {
                        return 0.0;
                    }
                    return idf * wdt / (wdt + K1);
                })
                .mapToDouble(x -> x)
                .sum();

        double pagerank = lambda * V(d.getPageRank());
        return bm25 + pagerank;
    }
}
