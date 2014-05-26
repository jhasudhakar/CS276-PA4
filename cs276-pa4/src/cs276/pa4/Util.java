package cs276.pa4;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class Util {
    private static double N = 98998;
    public static Map<Query, Map<String, Document>> loadTrainData(String feature_file_name) {
        File feature_file = new File(feature_file_name);
        if (!feature_file.exists()) {
            System.err.println("Invalid feature file name: " + feature_file_name);
            return null;
        }

        /* feature dictionary: Query -> (url -> Document)  */
        Map<Query, Map<String, Document>> queryDict = new HashMap<>();

        try {
            BufferedReader reader = new BufferedReader(new FileReader(feature_file));
            String line = null, url = null, anchor_text = null;
            Query currentQuery = null;
            Map<String, Document> currentDocuments = null;
            Document currentDocument = null;

            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(":", 2);
                String key = tokens[0].trim();
                String value = tokens[1].trim();

                if (key.equals("query")) {
                    currentQuery = new Query(value.trim());
                    currentDocuments = new HashMap<>();
                    queryDict.put(currentQuery, currentDocuments);
                } else if (key.equals("url")) {
                    url = value.trim();
                    currentDocument = new Document(url);
                    currentDocuments.put(url, currentDocument);
                } else if (key.equals("title")) {
                    currentDocument.setTitle(value);
                } else if (key.equals("header")) {
                    currentDocument.addHeader(value);
                } else if (key.equals("body_hits")) {
                    String[] temp = value.split(" ", 2);
                    String term = temp[0].trim();
                    List<Integer> positions = Arrays.asList(temp[1].trim().split(" "))
                            .stream()
                            .map(pos -> Integer.parseInt(pos))
                            .collect(Collectors.toList());

                    currentDocument.addBodyHits(term, positions);
                } else if (key.equals("body_length")) {
                    currentDocument.setBodyLength(Integer.parseInt(value));
                } else if (key.equals("pagerank")) {
                    currentDocument.setPageRank(Integer.parseInt(value));
                } else if (key.equals("anchor_text")) {
                    anchor_text = value.trim();
                } else if (key.equals("stanford_anchor_count")) {
                    currentDocument.addAnchor(anchor_text, Integer.parseInt(value));
                }
            }

            // finish build documents
            queryDict.values()
                    .stream()
                    .flatMap(ds -> ds.values().stream())
                    .forEach(d -> d.end());

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
            queryDict = null;
        }

        return queryDict;
    }

    public static Map<String, Double> loadDFs(String dfFile) throws IOException {
        Map<String, Double> dfs = new HashMap<>();

        BufferedReader br = new BufferedReader(new FileReader(dfFile));
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (line.equals("")) continue;
            String[] tokens = line.split("\\s+");
            dfs.put(tokens[0], Math.log(N / Double.parseDouble(tokens[1])));
        }
        br.close();
        return dfs;
    }

    /* query -> (url -> score) */
    public static Map<String, Map<String, Double>> loadRelData(String rel_file_name) {
        Map<String, Map<String, Double>> result = new HashMap<>();

        File rel_file = new File(rel_file_name);
        if (!rel_file.exists()) {
            System.err.println("Invalid feature file name: " + rel_file_name);
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(rel_file));
            String line = null, query = null, url = null;
            int numQuery = 0;
            int numDoc = 0;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(":", 2);
                String key = tokens[0].trim();
                String value = tokens[1].trim();

                if (key.equals("query")) {
                    query = value;
                    result.put(query, new HashMap<String, Double>());
                    numQuery++;
                } else if (key.equals("url")) {
                    String[] tmps = value.split(" ", 2);
                    url = tmps[0].trim();
                    double score = Double.parseDouble(tmps[1].trim());
                    result.get(query).put(url, score);
                    numDoc++;
                }
            }
            reader.close();
            System.err.println("# Rel file " + rel_file_name + ": number of queries=" + numQuery + ", number of documents=" + numDoc);
        } catch (Exception e) {
            e.printStackTrace();
            result = null;
        }

        return result;
    }

    public static void main(String[] args) {
        System.out.print(loadRelData(args[0]));
    }
}
