package cs276.pa4.util;

import java.io.*;

/**
 * Created by kavinyao on 5/16/14.
 */
public class SerializationHelper {
    public static boolean saveObjectToFile(Object o, String filename) {
        try {
            FileOutputStream fos = new FileOutputStream(filename);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(o);
            oos.close();

            return true;
        } catch (IOException e) {
            return false;
        }
    }

    public static Object loadObjectFromFile(String filename) {
        try {
            FileInputStream fis = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fis);
            Object o = ois.readObject();
            ois.close();

            return o;
        } catch(IOException | ClassNotFoundException ioe) {
            return null;
        }
    }
}
