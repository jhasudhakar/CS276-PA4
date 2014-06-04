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
    private static int NumFeatures = 9;
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
        attributes.add(new Attribute("url_w"));
        attributes.add(new Attribute("title_w"));
        attributes.add(new Attribute("body_w"));
        attributes.add(new Attribute("header_w"));
        attributes.add(new Attribute("anchor_w"));

        attributes.add(new Attribute("bm25_w"));
        attributes.add(new Attribute("window_w"));
        attributes.add(new Attribute("pagerank_w"));

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
    protected double[] extractFeaturesFromDocument(Query q, Document doc,
                                                   double score, Map<String, Double> idfs) {
        double[] tfidfFeatures = super.extractFeaturesFromDocument(q, doc, score, idfs);

        ArrayList<Double> features = new ArrayList<>();
        for (double feature : tfidfFeatures) {
            features.add(feature);
        }

        features.add(getSimScore(doc, q, idfs));
        HashSet<String> termSet = new HashSet<>(q.getQueryWords());
        features.add((double) doc.getSmallestWindow(termSet));
        features.add((double) doc.getPageRank());

        double[] fs = new double[NumFeatures];
        for (int i = 0; i < NumFeatures; ++i) {
            fs[i] = features.get(i);
        }

        return fs;
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
                    double idf = idfs.get(t);
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
