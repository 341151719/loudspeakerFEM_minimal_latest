import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class ComsolReq7Export {
  static final int[] PML_DOMAINS = {1, 5};
  static final double[] FREQS = {20, 50, 100, 200, 400, 500, 600, 1000, 1320, 2000, 5000, 6300, 8000};
  static final double[] LINE_FREQS = {50, 400, 500, 600, 1000, 2000, 6300};
  static int tagCounter = 0;

  public static void main(String[] args) throws Exception {
    try {
      if (args == null) args = new String[0];
      Locale.setDefault(Locale.US);
      String mph = args.length > 0 ? args[0] : "loudspeaker_driver_req2_solved.mph";
      String outDir = args.length > 1 ? args[1] : "comsol_req7_raw";
      new File(outDir).mkdirs();
      ModelUtil.initStandalone(false);
      Model model = ModelUtil.load("req7", mph);

      exportPmlProperties(model, outDir);
      exportTypicalWavelength(model, outDir);
      exportCoordinateJacobian(model, outDir);
      exportPressureLinecuts(model, outDir);
      exportExpressionProbe(model, outDir);
      writeReadme(outDir);
      System.out.println("REQ7 raw export complete: " + new File(outDir).getAbsolutePath());
    } catch (Throwable t) {
      try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("comsol_req7_exception.txt", StandardCharsets.UTF_8)))) {
        t.printStackTrace(pw);
      }
      t.printStackTrace();
      throw t;
    }
  }

  static String tag(String p) { tagCounter++; return p + "_" + tagCounter; }
  static String f(double v) { return (Double.isNaN(v) || Double.isInfinite(v)) ? "" : String.format(Locale.US, "%.15g", v); }
  static String csvq(String s) { return s == null ? "" : "\"" + s.replace("\"", "\"\"") + "\""; }

  static PrintWriter csv(String dir, String name, String header) throws Exception {
    PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(new File(dir, name), StandardCharsets.UTF_8)));
    pw.println(header);
    return pw;
  }

  static double[][][] global(Model m, String ds, String[] expr) {
    String t = tag("g");
    m.result().numerical().create(t, "EvalGlobal");
    m.result().numerical(t).set("data", ds);
    m.result().numerical(t).set("expr", expr);
    return new double[][][]{m.result().numerical(t).getReal(), m.result().numerical(t).getImag()};
  }

  static double[] freqs(Model m) {
    return global(m, "dset3", new String[]{"freq"})[0][0];
  }

  static int nearestSol(double[] fs, double target) {
    int best = 0; double err = Double.POSITIVE_INFINITY;
    for (int i = 0; i < fs.length; i++) {
      double e = Math.abs(fs[i] - target);
      if (e < err) { err = e; best = i; }
    }
    return best + 1;
  }

  static void exportPmlProperties(Model model, String dir) throws Exception {
    try (PrintWriter pw = csv(dir, "pml_feature_properties.csv", "node,property,value")) {
      dumpProperties(pw, "coordSystem.pml1", model.component("comp1").coordSystem("pml1"));
      dumpProperties(pw, "selection.sel8", model.component("comp1").selection("sel8"));
    }
    try (PrintWriter pw = csv(dir, "pml_source_model_snippet.csv", "source,line_no,line_text")) {
      pw.println("loudspeaker_driver.java,85," + csvq("model.component(\"comp1\").selection().create(\"sel8\", \"Explicit\");"));
      pw.println("loudspeaker_driver.java,87," + csvq("model.component(\"comp1\").selection(\"sel8\").set(1, 5);"));
      pw.println("loudspeaker_driver.java,112," + csvq("model.component(\"comp1\").coordSystem().create(\"pml1\", \"PML\");"));
      pw.println("loudspeaker_driver.java,113," + csvq("model.component(\"comp1\").coordSystem(\"pml1\").selection().named(\"sel8\");"));
      pw.println("loudspeaker_driver.java,114," + csvq("model.component(\"comp1\").coordSystem(\"pml1\").set(\"PMLgamma\", \"3\");"));
    }
  }

  static void dumpProperties(PrintWriter pw, String node, Object obj) {
    try {
      String[] props = (String[]) obj.getClass().getMethod("properties").invoke(obj);
      for (String key : props) {
        String val;
        try {
          Object v = obj.getClass().getMethod("getString", String.class).invoke(obj, key);
          val = String.valueOf(v);
        } catch (Throwable t1) {
          try {
            Object v = obj.getClass().getMethod("get", String.class).invoke(obj, key);
            val = Arrays.deepToString(new Object[]{v});
          } catch (Throwable t2) {
            val = "<unreadable:" + t2.getClass().getSimpleName() + ">";
          }
        }
        pw.println(csvq(node) + "," + csvq(key) + "," + csvq(val));
      }
    } catch (Throwable t) {
      pw.println(csvq(node) + ",ERROR," + csvq(t.toString()));
    }
  }

  static void exportTypicalWavelength(Model model, String dir) throws Exception {
    double[] fs = freqs(model);
    try (PrintWriter pw = csv(dir, "pml_typical_wavelength_vs_frequency.csv",
      "requested_freq_Hz,solved_freq_Hz,solution_index,sound_speed_m_s,wavenumber_rad_m,typical_wavelength_m,pml_scaling_factor,pml_gamma,effective_pml_scale_m")) {
      for (double ft : FREQS) {
        int sol = nearestSol(fs, ft);
        String t = tag("tw");
        model.result().numerical().create(t, "Eval");
        model.result().numerical(t).set("data", "dset3");
        model.result().numerical(t).selection().set(1);
        model.result().numerical(t).set("expr", new String[]{"acpr.c", "acpr.k*1[m]", "(2*pi/acpr.k)/1[m]", "pml1.PMLgamma"});
        model.result().numerical(t).set("solnum", sol);
        double[][] re = model.result().numerical(t).getReal();
        double c = re[0][0], k = re[0][1], lambda = re[0][2], gamma = re[0][3];
        double factor = 1.0;
        pw.println(String.join(",", f(ft), f(fs[sol - 1]), Integer.toString(sol), f(c), f(k), f(lambda), f(factor), f(gamma), f(lambda * factor)));
      }
    }
  }

  static void exportCoordinateJacobian(Model model, String dir) throws Exception {
    double[] fs = freqs(model);
    String[] expr = {
      "r/1[m]", "z/1[m]",
      "real(pml1.r/1[m])", "imag(pml1.r/1[m])", "real(pml1.z/1[m])", "imag(pml1.z/1[m])",
      "d(real(pml1.r/1[m]),r)", "d(imag(pml1.r/1[m]),r)", "d(real(pml1.r/1[m]),z)", "d(imag(pml1.r/1[m]),z)",
      "d(real(pml1.z/1[m]),r)", "d(imag(pml1.z/1[m]),r)", "d(real(pml1.z/1[m]),z)", "d(imag(pml1.z/1[m]),z)",
      "acpr.p_t", "d(acpr.p_t,r)", "d(acpr.p_t,z)", "acpr.c", "acpr.k"
    };
    try (PrintWriter pw = csv(dir, "pml_coordinate_jacobian_points.csv",
      "requested_freq_Hz,solved_freq_Hz,solution_index,domain_id,node_id,r_m,z_m,r_stretched_real_m,r_stretched_imag_m,z_stretched_real_m,z_stretched_imag_m,J11_real,J11_imag,J12_real,J12_imag,J21_real,J21_imag,J22_real,J22_imag,detJ_real,detJ_imag,invJ11_real,invJ11_imag,invJ12_real,invJ12_imag,invJ21_real,invJ21_imag,invJ22_real,invJ22_imag,r_stretched_over_r_real,r_stretched_over_r_imag,axisym_volume_factor_real,axisym_volume_factor_imag,p_real,p_imag,dpdr_real,dpdr_imag,dpdz_real,dpdz_imag,sound_speed_m_s,wavenumber_rad_m")) {
      for (double ft : FREQS) {
        int sol = nearestSol(fs, ft);
        for (int dom : PML_DOMAINS) {
          try {
            String t = tag("jac");
            model.result().numerical().create(t, "Eval");
            model.result().numerical(t).set("data", "dset3");
            model.result().numerical(t).selection().set(dom);
            model.result().numerical(t).set("expr", expr);
            model.result().numerical(t).set("solnum", sol);
            double[][] re = model.result().numerical(t).getReal();
            double[][] im = model.result().numerical(t).getImag();
            int step = Math.max(1, re.length / 500);
            int nid = 0;
            for (int i = 0; i < re.length; i += step) {
              nid++;
              double r = re[i][0], z = re[i][1];
              double rsr = re[i][2], rsi = re[i][3], zsr = re[i][4], zsi = re[i][5];
              double a = re[i][6], ai = re[i][7], b = re[i][8], bi = re[i][9];
              double c = re[i][10], ci = re[i][11], d = re[i][12], di = re[i][13];
              double detR = a*d - ai*di - b*c + bi*ci;
              double detI = a*di + ai*d - b*ci - bi*c;
              double den = detR*detR + detI*detI;
              double inv11R = (d*detR + di*detI)/den, inv11I = (di*detR - d*detI)/den;
              double inv12R = (-b*detR - bi*detI)/den, inv12I = (-bi*detR + b*detI)/den;
              double inv21R = (-c*detR - ci*detI)/den, inv21I = (-ci*detR + c*detI)/den;
              double inv22R = (a*detR + ai*detI)/den, inv22I = (ai*detR - a*detI)/den;
              double ratioR = r == 0 ? Double.NaN : rsr / r;
              double ratioI = r == 0 ? Double.NaN : rsi / r;
              double volR = detR * ratioR - detI * ratioI;
              double volI = detR * ratioI + detI * ratioR;
              pw.println(String.join(",", f(ft), f(fs[sol - 1]), Integer.toString(sol), Integer.toString(dom), Integer.toString(nid), f(r), f(z), f(rsr), f(rsi), f(zsr), f(zsi),
                f(a), f(ai), f(b), f(bi), f(c), f(ci), f(d), f(di), f(detR), f(detI),
                f(inv11R), f(inv11I), f(inv12R), f(inv12I), f(inv21R), f(inv21I), f(inv22R), f(inv22I),
                f(ratioR), f(ratioI), f(volR), f(volI), f(re[i][14]), f(im[i][14]), f(re[i][15]), f(im[i][15]), f(re[i][16]), f(im[i][16]), f(re[i][17]), f(re[i][18])));
            }
          } catch (Throwable ignored) {}
        }
      }
    }
  }

  static void exportPressureLinecuts(Model model, String dir) throws Exception {
    double[] fs = freqs(model);
    String[] expr = {"r/1[m]", "z/1[m]", "pml1.r/1[m]", "pml1.z/1[m]", "acpr.p_t", "d(acpr.p_t,r)", "d(acpr.p_t,z)"};
    try (PrintWriter pw = csv(dir, "pml_pressure_linecuts.csv",
      "requested_freq_Hz,solved_freq_Hz,solution_index,domain_id,node_id,r_m,z_m,pml_r_real_m,pml_r_imag_m,pml_z_real_m,pml_z_imag_m,p_real,p_imag,p_abs,dpdr_real,dpdr_imag,dpdz_real,dpdz_imag,depth_proxy,depth_bin")) {
      for (double ft : LINE_FREQS) {
        int sol = nearestSol(fs, ft);
        for (int dom : PML_DOMAINS) {
          String t = tag("line");
          model.result().numerical().create(t, "Eval");
          model.result().numerical(t).set("data", "dset3");
          model.result().numerical(t).selection().set(dom);
          model.result().numerical(t).set("expr", expr);
          model.result().numerical(t).set("solnum", sol);
          double[][] re = model.result().numerical(t).getReal();
          double[][] im = model.result().numerical(t).getImag();
          double minR = Double.POSITIVE_INFINITY, maxR = Double.NEGATIVE_INFINITY;
          for (double[] row : re) { minR = Math.min(minR, row[0]); maxR = Math.max(maxR, row[0]); }
          int step = Math.max(1, re.length / 1000);
          int nid = 0;
          for (int i = 0; i < re.length; i += step) {
            nid++;
            double depth = (maxR == minR) ? 0 : (re[i][0] - minR) / (maxR - minR);
            String bin = depth < 0.125 ? "inner" : depth < 0.375 ? "p25" : depth < 0.625 ? "p50" : depth < 0.875 ? "p75" : "outer";
            double pr = re[i][4], pi = im[i][4];
            pw.println(String.join(",", f(ft), f(fs[sol - 1]), Integer.toString(sol), Integer.toString(dom), Integer.toString(nid),
              f(re[i][0]), f(re[i][1]), f(re[i][2]), f(im[i][2]), f(re[i][3]), f(im[i][3]),
              f(pr), f(pi), f(Math.hypot(pr, pi)), f(re[i][5]), f(im[i][5]), f(re[i][6]), f(im[i][6]), f(depth), bin));
          }
        }
      }
    }
  }

  static void exportExpressionProbe(Model model, String dir) throws Exception {
    String[] candidates = {"pml1.r", "pml1.z", "pml1.PMLgamma", "acpr.c", "acpr.k", "acpr.rho", "d(pml1.r,r)", "d(pml1.r,z)", "d(pml1.z,r)", "d(pml1.z,z)"};
    try (PrintWriter pw = csv(dir, "pml_expression_probe.csv", "expression,status,real_sample,imag_sample,notes")) {
      for (String e : candidates) {
        try {
          String t = tag("probe");
          model.result().numerical().create(t, "Eval");
          model.result().numerical(t).set("data", "dset3");
          model.result().numerical(t).selection().set(1);
          model.result().numerical(t).set("expr", new String[]{e});
          model.result().numerical(t).set("solnum", nearestSol(freqs(model), 6300));
          double[][] re = model.result().numerical(t).getReal();
          double[][] im = model.result().numerical(t).getImag();
          pw.println(csvq(e) + ",ok," + f(re[0][0]) + "," + f(im[0][0]) + ",");
        } catch (Throwable t) {
          pw.println(csvq(e) + "," + csvq("failed") + ",,," + csvq(t.toString()));
        }
      }
    }
  }

  static void writeReadme(String dir) throws Exception {
    try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(new File(dir, "README_REQ7_RAW.txt"), StandardCharsets.UTF_8)))) {
      pw.println("REQ7 raw export");
      pw.println("Purpose: audit COMSOL PML settings, typical wavelength, stretched coordinates/Jacobian, axisymmetric radial factor, and PML pressure fields.");
      pw.println("PML coordinate-system tag: pml1. Domains: 1,5 via selection sel8.");
      pw.println("COMSOL exposes pml1.r and pml1.z but not explicit J11/detJ variables; Jacobian columns are evaluated as derivatives d(pml1.r,z/r), d(pml1.z,z/r).");
      pw.println("pml_pressure_linecuts.csv uses a radius-based depth_proxy over each PML domain to label inner/p25/p50/p75/outer samples.");
    }
  }
}
