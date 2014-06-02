package cs276.pa4.util;

/**
 * Created by kavinyao on 6/1/14.
 */
public interface BiFunction<F, S, R> {
    R apply(F first, S second);
}
