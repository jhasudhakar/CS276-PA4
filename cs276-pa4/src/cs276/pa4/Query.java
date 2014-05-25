package cs276.pa4;

import cs276.pa4.doc.FieldProcessor;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

public class Query {
    private String originalQuery;
    private List<String> queryWords;

    public Query(String query) {
        originalQuery = query;
        // remove duplicates (why use LinkedHashSet?)
        queryWords = new ArrayList<>(new LinkedHashSet<>(FieldProcessor.splitField(query)));
    }

    public List<String> getQueryWords() {
        return queryWords;
    }

    public String getOriginalQuery() {
        return originalQuery;
    }

    @Override
    public String toString() {
        return "Query<" + originalQuery + ">";
    }

    @Override
    public int hashCode() {
        return originalQuery.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }

        if (obj == null || !(obj instanceof Query)) {
            return false;
        }

        Query another = (Query)obj;
        return this.originalQuery.equals(another.originalQuery);
    }
}
