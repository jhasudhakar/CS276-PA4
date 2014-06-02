package cs276.pa4.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by kavinyao on 6/1/14.
 */
public class ListUtility {
    public static <F, T> List<T> map(List<F> list, UnaryFunction<F, T> op) {
        List<T> result = new ArrayList<T>();

        for (F f : list) {
            result.add(op.apply(f));
        }

        return result;
    }

    public static <T> List<T> filter(List<T> list, Predicate<T> pred) {
        List<T> result = new ArrayList<T>();

        for (T t : list) {
            if (pred.test(t)) {
                result.add(t);
            }
        }

        return result;
    }

    public static int sum(List<Integer> numbers) {
        int sum = 0;
        for (Integer number : numbers) {
            sum += number;
        }
        return sum;
    }
}
