import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class ComsolReq6VariantRerun {
  static final double[] FREQS = {100, 1000, 6300};
  static final double[] RADII = {1, 2, 10};
  static int tagCounter = 0;

  public static void main(String[] args) throws Exception {
    try {
      Locale.setDefault(Locale.US);
      File out = new File("comsol_req6_raw");
      out.mkdirs();
      ModelUtil.initStandalone(false);
      Model model = ModelUtil.load("req6var", "loudspeaker_driver_req2_solved.mph");

      try (PrintWriter pw = csv(out, "req6_cloned_variant_rerun_status.csv", "step,status,notes")) {
        pw.println("load,ok,loudspeaker_driver_req2_solved.mph");
        duplicateVariant(model, "efc2", "pext_nosym", false, false, pw);
        duplicateVariant(model, "efc4", "pext_rev", true, true, pw);
        try {
          model.study("std2").run();
          pw.println("study_std2_run,ok,recomputed after cloned EFC variants were added");
        } catch (Throwable t) {
          pw.println("study_std2_run," + csvq("failed:" + t.toString()) + ",variant functions may remain unavailable");
        }
      }

      exportPext(model, out);
      System.out.println("REQ6 cloned variant rerun export complete");
    } catch (Throwable t) {
      try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("comsol_req6_variant_exception.txt", StandardCharsets.UTF_8)))) {
        t.printStackTrace(pw);
      }
      t.printStackTrace();
      throw t;
    }
  }

  static PrintWriter csv(File dir, String name, String header) throws Exception {
    PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(new File(dir, name), StandardCharsets.UTF_8)));
    pw.println(header);
    return pw;
  }

  static String f(double v) { return (Double.isNaN(v) || Double.isInfinite(v)) ? "" : String.format(Locale.US, "%.15g", v); }
  static String csvq(String s) { return s == null ? "" : "\"" + s.replace("\"", "\"\"") + "\""; }
  static double phase(double re, double im) { return Math.atan2(im, re) * 180.0 / Math.PI; }
  static String tag(String prefix) { tagCounter++; return prefix + "_" + tagCounter; }

  static void duplicateVariant(Model model, String tag, String ext, boolean keepSym, boolean reverse, PrintWriter pw) {
    try {
      model.component("comp1").physics("acpr").feature().duplicate(tag, "efc1");
      model.component("comp1").physics("acpr").feature(tag).set("ExtName", ext);
      model.component("comp1").physics("acpr").feature(tag).setIndex("SymmetryCondition0", 0, 0);
      model.component("comp1").physics("acpr").feature(tag).setIndex("SymmetryCondition2", keepSym ? 1 : 0, 0);
      model.component("comp1").physics("acpr").feature(tag).setIndex("ReverseNormal", reverse ? 1 : 0, 0);
      pw.println("duplicate_" + tag + ",ok," + ext);
    } catch (Throwable t) {
      pw.println("duplicate_" + tag + "," + csvq("failed:" + t.toString()) + "," + ext);
    }
  }

  static double[][][] global(Model m, String ds, String[] expr, int sol) {
    String t = tag("g");
    m.result().numerical().create(t, "EvalGlobal");
    m.result().numerical(t).set("data", ds);
    m.result().numerical(t).set("expr", expr);
    m.result().numerical(t).set("solnum", sol);
    return new double[][][]{m.result().numerical(t).getReal(), m.result().numerical(t).getImag()};
  }

  static double[] freqs(Model m) {
    String t = tag("freq");
    m.result().numerical().create(t, "EvalGlobal");
    m.result().numerical(t).set("data", "dset3");
    m.result().numerical(t).set("expr", new String[]{"freq"});
    return m.result().numerical(t).getReal()[0];
  }

  static int nearestSol(double[] fs, double target) {
    int best = 0; double err = Double.POSITIVE_INFINITY;
    for (int i = 0; i < fs.length; i++) {
      double e = Math.abs(fs[i] - target);
      if (e < err) { err = e; best = i; }
    }
    return best + 1;
  }

  static void exportPext(Model model, File out) throws Exception {
    double[] fs = freqs(model);
    String[][] variants = {
      {"original_rerun", "pext", "efc1"},
      {"no_symmetry_cloned_rerun", "pext_nosym", "efc2"},
      {"reverse_normal_cloned_rerun", "pext_rev", "efc4"}
    };
    try (PrintWriter pw = csv(out, "req6_pext_cloned_variant_rerun_complex.csv",
      "variant_label,feature_tag,ext_name,requested_freq_Hz,solved_freq_Hz,solution_index,radius_m,theta_deg,eval_r_m,eval_z_m,pext_real_Pa,pext_imag_Pa,pext_abs_Pa,pext_phase_deg,SPL_abs_dB,SPL_relative_to_0deg_dB,amp_times_R,status")) {
      for (String[] v : variants) {
        for (double ft : FREQS) {
          int sol = nearestSol(fs, ft);
          for (double R : RADII) {
            String[] expr = new String[181];
            for (int k = 0; k < 181; k++) {
              int theta = -90 + k;
              double rad = Math.toRadians(theta);
              expr[k] = String.format(Locale.US, "%s(%.12g[m],%.12g[m])", v[1], Math.abs(R * Math.sin(rad)), R * Math.cos(rad));
            }
            try {
              double[][][] ri = global(model, "dset3", expr, sol);
              double[][] re = ri[0], im = ri[1];
              double p0 = Math.hypot(re[90][0], im[90][0]);
              double spl0 = 20 * Math.log10(p0 / 20e-6);
              for (int k = 0; k < 181; k++) {
                int theta = -90 + k;
                double rad = Math.toRadians(theta);
                double rr = Math.abs(R * Math.sin(rad));
                double zz = R * Math.cos(rad);
                double pr = re[k][0], pi = im[k][0], pa = Math.hypot(pr, pi);
                double spl = 20 * Math.log10(pa / 20e-6);
                pw.println(String.join(",", csvq(v[0]), v[2], v[1], f(ft), f(fs[sol - 1]), Integer.toString(sol), f(R), Integer.toString(theta), f(rr), f(zz), f(pr), f(pi), f(pa), f(phase(pr, pi)), f(spl), f(spl - spl0), f(pa * R), "ok"));
              }
            } catch (Throwable t) {
              pw.println(csvq(v[0]) + "," + v[2] + "," + v[1] + "," + f(ft) + "," + f(fs[sol - 1]) + "," + sol + "," + f(R) + ",ERROR,,,,,,,,,," + csvq("failed:" + t.toString()));
            }
          }
        }
      }
    }
  }
}
