import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class ComsolReq6Probe {
  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.US);
    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("req6probe", "loudspeaker_driver_req2_solved.mph");
    try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("comsol_req6_probe_stdout.txt", StandardCharsets.UTF_8)))) {
      dump(pw, "acpr.efc1", model.component("comp1").physics("acpr").feature("efc1"));
      dump(pw, "pg10.dir1", model.result("pg10").feature("dir1"));
      dump(pw, "pg7.oct1", model.result("pg7").feature("oct1"));
    }
  }

  static void dump(PrintWriter pw, String name, Object obj) {
    pw.println("## " + name);
    try {
      java.lang.reflect.Method props = obj.getClass().getMethod("properties");
      Object p = props.invoke(obj);
      pw.println("properties=" + Arrays.deepToString(new Object[]{p}));
      if (p instanceof String[]) {
        for (String key : (String[]) p) {
          try {
            java.lang.reflect.Method getString = obj.getClass().getMethod("getString", String.class);
            Object v = getString.invoke(obj, key);
            pw.println(key + "=" + v);
          } catch (Throwable t1) {
            try {
              java.lang.reflect.Method get = obj.getClass().getMethod("get", String.class);
              Object v = get.invoke(obj, key);
              pw.println(key + "=" + Arrays.deepToString(new Object[]{v}));
            } catch (Throwable t2) {
              pw.println(key + "=<unreadable:" + t2.getClass().getSimpleName() + ">");
            }
          }
        }
      }
    } catch (Throwable t) {
      pw.println("ERROR " + t);
      for (java.lang.reflect.Method m : obj.getClass().getMethods()) {
        if (m.getName().toLowerCase(Locale.US).contains("propert") || m.getName().equals("get") || m.getName().startsWith("get")) {
          pw.println("method " + m);
        }
      }
    }
  }
}
