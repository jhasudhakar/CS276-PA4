package cs276.pa4;

import cs276.pa4.doc.DocField;
import cs276.pa4.util.ListUtility;
import cs276.pa4.util.MapUtility;
import cs276.pa4.util.UnaryFunction;
import weka.core.Attribute;

import java.util.*;

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

        attributes.add(new Attribute("bm25_w"));
        attributes.add(new Attribute("window_w"));
        attributes.add(new Attribute("pagerank_w"));

        attributes.add(new Attribute("fields_w"));

        return attributes;
    }

    /**
     * Calculate average lengths of document fields.
     * @param queryDict
     */
    @Override
    protected void processDocuments(Map<Query, Map<String, Document>> queryDict) {
        Set<Document> uniqueDocSet = new HashSet<>();
        for (Map<String, Document> docs : queryDict.values()) {
            uniqueDocSet.addAll(docs.values());
        }
        List<Document> uniqueDocs = new ArrayList<>(uniqueDocSet);

        // compute length of each field
        lengths = new HashMap<>();
        for (DocField f : DocField.values()) {
            // for closure to work in Java 7
            final DocField ff = f;
            lengths.put(f, lengthsOfField(uniqueDocs, new UnaryFunction<Document, Integer>() {
                @Override
                public Integer apply(Document d) {
                    return d.getNumFieldTokens(ff);
                }
            }));
        }

        // compute average lengths of each field
        avgLengths = new HashMap<>();
        for (DocField docField : DocField.values()) {
            avgLengths.put(docField, averageFieldLength(lengths.get(docField)));
            //System.err.println("avg(" + docField + ") = " + avgLengths.get(docField));
        }

        pagerankScores = new HashMap<>();
        for (Document doc : uniqueDocs) {
            pagerankScores.put(doc, (double) doc.getPageRank());
        }
    }

    /**
     * Compute length of given field for every document.
     * @param docs must not contain duplicate
     * @param getLength length getter for field
     * @return
     */
    private static Map<Document, Double> lengthsOfField(List<Document> docs,
                                                        UnaryFunction<Document, Integer> getLength) {
        Map<Document, Double> lengths = new HashMap<>();
        for (Document doc : docs) {
            lengths.put(doc, getLength.apply(doc).doubleValue());
        }
        return lengths;
    }

    private static Double averageFieldLength(Map<Document, Double> fieldLengths) {
        double sum = sum(fieldLengths.values());
        return fieldLengths.size() == 0 ? 0.0 : sum / fieldLengths.size();
    }

    private static double sum(Collection<Double> fieldLengths) {
        double sum = 0.0;
        for (Double l : fieldLengths) {
            sum += l;
        }
        return sum;
    }

    private double getTermWeight(final Document d, final Map<DocField, Map<String, Double>> tfs, final String t, Query q) {
        return sum(ListUtility.map(Arrays.asList(DocField.values()), new UnaryFunction<DocField, Double>() {
            @Override
            public Double apply(DocField f) {
                double tf = MapUtility.getWithFallback(tfs.get(f), t, 0.0);
                double denominator = 1 + Bf.get(f) * (lengths.get(f).get(d) / avgLengths.get(f) - 1);
                double ftf = denominator == 0.0 ? 0.0 : tf / denominator;
                return Wf.get(f) * ftf;
            }
        }));
    }

    @Override
    protected void flip(double[] diffFS) {
        super.flip(diffFS);

//        for (int k = 15; k < 19; ++k) {
//            diffFS[k] = diffFS[k] == 0.0 ? 0.0 : 1.0;
//        }
    }

    @Override
    protected double[] getFSDiff(double[] fs1, double[] fs2) {
        double[] diffFS = super.getFSDiff(fs1, fs2);

//        for (int k = 15; k < 19; ++k) {
//            diffFS[k] = diffFS[k] == 0.0 ? 0.0 : 1.0;
//        }

        return diffFS;
    }

    @Override
    protected double[] extractFeaturesFromDocument(Query q, Document doc,
                                                   double score, Map<String, Double> idfs) {
        Map<DocField, Map<String, Double>> tfs = getRawDocTermFreqs(q, doc);
        Map<String, Double> tfQuery = getQueryFreqs(q, idfs);

        double[] instance = new double[10];

        // tf-idf features
        instance[0] = score;
        instance[1] = dotProduct(tfQuery, tfs.get(DocField.url));
        instance[2] = dotProduct(tfQuery, tfs.get(DocField.title));
        instance[3] = dotProduct(tfQuery, tfs.get(DocField.body));
        instance[4] = dotProduct(tfQuery, tfs.get(DocField.header));
        instance[5] = dotProduct(tfQuery, tfs.get(DocField.anchor));

        // extended features
        instance[6] = getSimScore(doc, q, idfs);

        HashSet<String> termSet = new HashSet<>(q.getQueryWords());
        instance[7] = doc.getSmallestWindow(termSet);
        instance[8] = doc.getPageRank();

        int num = 0;
        for (int i = 1; i < 6; i++) {
            if (instance[i] > 0) {
                num++;
            }
        }
        instance[9] = num;

        return instance;
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

    private double getSimScore(final Document d, final Query q, final Map<String, Double> idfs) {
        final Map<DocField, Map<String, Double>> tfs = getRawDocTermFreqs(q, d);
        Map<String, Double> tfQuery = getQueryFreqs(q, idfs);

        double bm25 = sum(ListUtility.map(new ArrayList<>(tfQuery.keySet()), new UnaryFunction<String, Double>() {
            @Override
            public Double apply(String t) {
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
            }
        }));

        double pagerank = lambda * V(d.getPageRank());
        return bm25 + pagerank;
    }
}
