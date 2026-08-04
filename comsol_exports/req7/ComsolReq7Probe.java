import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class ComsolReq7Probe {
  static int tagCounter = 0;
  public static void main(String[] args) throws Exception {
    Locale.setDefault(Locale.US);
    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("req7probe", "loudspeaker_driver_req2_solved.mph");
    try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("comsol_req7_probe_stdout.txt", StandardCharsets.UTF_8)))) {
      dumpProperties(pw, "coordSystem.pml1", model.component("comp1").coordSystem("pml1"));
      dumpProperties(pw, "selection.sel8", model.component("comp1").selection("sel8"));
      probeExpressions(model, pw);
    }
  }

  static String tag(String p) { tagCounter++; return p + "_" + tagCounter; }
  static String csvq(String s) { return s == null ? "" : "\"" + s.replace("\"", "\"\"") + "\""; }

  static void dumpProperties(PrintWriter pw, String name, Object obj) {
    pw.println("## " + name);
    try {
      String[] props = (String[]) obj.getClass().getMethod("properties").invoke(obj);
      pw.println("properties=" + Arrays.toString(props));
      for (String key : props) {
        try {
          Object v = obj.getClass().getMethod("getString", String.class).invoke(obj, key);
          pw.println(key + "=" + v);
        } catch (Throwable t1) {
          try {
            Object v = obj.getClass().getMethod("get", String.class).invoke(obj, key);
            pw.println(key + "=" + Arrays.deepToString(new Object[]{v}));
          } catch (Throwable t2) {
            pw.println(key + "=<unreadable:" + t2.getClass().getSimpleName() + ">");
          }
        }
      }
    } catch (Throwable t) {
      pw.println("ERROR " + t);
    }
  }

  static void probeExpressions(Model model, PrintWriter pw) {
    String[] candidates = {
      "pml1.r", "pml1.z", "pml1.rp", "pml1.zp", "pml1.r_pml", "pml1.z_pml",
      "pml1.x", "pml1.y", "pml1.X", "pml1.Y", "pml1.X1", "pml1.X2",
      "pml1.s1", "pml1.s2", "pml1.sDist1", "pml1.sDist2", "pml1.dist1", "pml1.dist2",
      "pml1.xi1", "pml1.xi2", "pml1.eta1", "pml1.eta2",
      "pml1.J11", "pml1.J12", "pml1.J21", "pml1.J22", "pml1.detJ",
      "pml1.invJ11", "pml1.invJ12", "pml1.invJ21", "pml1.invJ22",
      "pml1.dx_dX", "pml1.dy_dY", "pml1.sx", "pml1.sy", "pml1.sz",
      "pml1.lambda", "pml1.k", "pml1.gamma", "pml1.PMLgamma",
      "acpr.c", "acpr.k", "acpr.k0", "acpr.rho", "acpr.rho_c", "acpr.cp", "c0", "343[m/s]/freq"
    };
    pw.println("## expression_probe");
    for (String expr : candidates) {
      try {
        String t = tag("probe");
        model.result().numerical().create(t, "Eval");
        model.result().numerical(t).set("data", "dset3");
        model.result().numerical(t).selection().set(1);
        model.result().numerical(t).set("expr", new String[]{expr});
        model.result().numerical(t).set("solnum", 122);
        double[][] re = model.result().numerical(t).getReal();
        double[][] im = model.result().numerical(t).getImag();
        pw.println(expr + ",OK," + re.length + "," + re[0][0] + "," + im[0][0]);
      } catch (Throwable t) {
        pw.println(expr + ",FAIL," + t.getClass().getSimpleName() + "," + t.getMessage().replace('\n', ' '));
      }
    }
  }
}
