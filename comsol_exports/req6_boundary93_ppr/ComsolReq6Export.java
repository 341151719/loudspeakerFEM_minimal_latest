import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class ComsolReq6Export {
  static final int B93 = 93;
  static final double[] FREQS = {100, 1000, 6300};
  static final double[] RADII = {1, 2, 10};
  static int tagCounter = 0;

  public static void main(String[] args) throws Exception {
    try {
      if (args == null) args = new String[0];
      Locale.setDefault(Locale.US);
      String mph = args.length > 0 ? args[0] : "loudspeaker_driver_req2_solved.mph";
      String outDir = args.length > 1 ? args[1] : "comsol_req6_raw";
      new File(outDir).mkdirs();
      ModelUtil.initStandalone(false);
      Model model = ModelUtil.load("req6", mph);

      exportProperties(model, outDir);
      ArrayList<String[]> variants = setupVariants(model);
      exportVariantStatus(outDir, variants);
      exportPextRadiusTheta(model, outDir, variants);
      exportOriginalDirectivityPlot(model, outDir);
      exportBoundary93GradientAudit(model, outDir);
      writeReadme(outDir);
      System.out.println("REQ6 raw export complete: " + new File(outDir).getAbsolutePath());
    } catch (Throwable t) {
      try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("comsol_req6_exception.txt", StandardCharsets.UTF_8)))) {
        t.printStackTrace(pw);
      }
      t.printStackTrace();
      throw t;
    }
  }

  static String tag(String prefix) { tagCounter++; return prefix + "_" + tagCounter; }
  static String f(double v) { return (Double.isNaN(v) || Double.isInfinite(v)) ? "" : String.format(Locale.US, "%.15g", v); }
  static String csvq(String s) { return s == null ? "" : "\"" + s.replace("\"", "\"\"") + "\""; }
  static double phase(double re, double im) { return Math.atan2(im, re) * 180.0 / Math.PI; }

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

  static double[] freqs(Model m, String ds) {
    return global(m, ds, new String[]{"freq"})[0][0];
  }

  static int nearestSol(double[] fs, double target) {
    int best = 0; double err = Double.POSITIVE_INFINITY;
    for (int i = 0; i < fs.length; i++) {
      double e = Math.abs(fs[i] - target);
      if (e < err) { err = e; best = i; }
    }
    return best + 1;
  }

  static void exportProperties(Model model, String dir) throws Exception {
    try (PrintWriter pw = csv(dir, "req6_efc_and_plot_properties.csv", "node,property,value")) {
      dumpProperties(pw, "acpr.efc1", model.component("comp1").physics("acpr").feature("efc1"));
      dumpProperties(pw, "result.pg10.dir1", model.result("pg10").feature("dir1"));
      dumpProperties(pw, "result.pg7.oct1", model.result("pg7").feature("oct1"));
    }
    try (PrintWriter pw = csv(dir, "req6_efc_source_from_model_script.csv", "source,line_no,line_text")) {
      pw.println("loudspeaker_driver.java,440," + csvq("model.component(\"comp1\").physics(\"acpr\").create(\"efc1\", \"ExteriorFieldCalculation\", 1);"));
      pw.println("loudspeaker_driver.java,441," + csvq("model.component(\"comp1\").physics(\"acpr\").feature(\"efc1\").selection().set(93);"));
      pw.println("loudspeaker_driver.java,442," + csvq("model.component(\"comp1\").physics(\"acpr\").feature(\"efc1\").setIndex(\"SymmetryCondition2\", 1, 0);"));
      pw.println("loudspeaker_driver.java,652," + csvq("model.result(\"pg7\").feature(\"oct1\").set(\"expr\", \"pext(0,1[m])\");"));
      pw.println("loudspeaker_driver.java,710," + csvq("model.result(\"pg10\").create(\"dir1\", \"Directivity\");"));
      pw.println("loudspeaker_driver.java,716," + csvq("model.result(\"pg10\").feature(\"dir1\").set(\"radius\", \"1[m]\");"));
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

  static ArrayList<String[]> setupVariants(Model model) {
    ArrayList<String[]> vars = new ArrayList<String[]>();
    vars.add(new String[]{"efc1", "pext", "original", "ok_existing"});
    vars.add(new String[]{"efc1", "pext", "no_symmetry", "toggle_efc1_SymmetryCondition0_0_SymmetryCondition2_0"});
    vars.add(new String[]{"efc1", "pext", "reverse_normal", "toggle_efc1_ReverseNormal_1"});
    vars.add(new String[]{"efc1", "pext", "farfield_approximation", "attempt_toggle_efc1_IntType_farfield_candidate_values"});
    return vars;
  }

  static void exportVariantStatus(String dir, ArrayList<String[]> vars) throws Exception {
    try (PrintWriter pw = csv(dir, "req6_operator_variant_status.csv", "variant_label,feature_tag,ext_name,status")) {
      for (String[] v : vars) pw.println(csvq(v[2]) + "," + v[0] + "," + v[1] + "," + csvq(v[3]));
    }
  }

  static void exportPextRadiusTheta(Model model, String dir, ArrayList<String[]> vars) throws Exception {
    double[] fs = freqs(model, "dset3");
    try (PrintWriter vpw = csv(dir, "req6_operator_variant_effective_properties.csv", "variant_label,property,value,status")) {
    try (PrintWriter pw = csv(dir, "req6_pext_radius_theta_complex.csv",
      "variant_label,feature_tag,ext_name,requested_freq_Hz,solved_freq_Hz,solution_index,radius_m,theta_deg,eval_r_m,eval_z_m,pext_real_Pa,pext_imag_Pa,pext_abs_Pa,pext_phase_deg,SPL_abs_dB,SPL_relative_to_0deg_dB,amp_times_R")) {
      for (String[] v : vars) {
        String applyStatus = applyVariantToEfc1(model, v[2]);
        vpw.println(csvq(v[2]) + ",apply_status," + csvq(applyStatus) + "," + csvq(applyStatus));
        dumpEffective(vpw, model, v[2], "UsePPR");
        dumpEffective(vpw, model, v[2], "ReverseNormal");
        dumpEffective(vpw, model, v[2], "ExtName");
        dumpEffective(vpw, model, v[2], "SymmetryCondition0");
        dumpEffective(vpw, model, v[2], "SymmetryCondition2");
        dumpEffective(vpw, model, v[2], "IntType");
        if (applyStatus.startsWith("failed")) {
          continue;
        }
        for (double ft : FREQS) {
          int sol = nearestSol(fs, ft);
          for (double R : RADII) {
            String[] expr = new String[181];
            int idx0 = -1;
            for (int k = 0; k < 181; k++) {
              int theta = -90 + k;
              if (theta == 0) idx0 = k;
              double rad = Math.toRadians(theta);
              double rr = Math.abs(R * Math.sin(rad));
              double zz = R * Math.cos(rad);
              expr[k] = String.format(Locale.US, "%s(%.12g[m],%.12g[m])", v[1], rr, zz);
            }
            try {
              String t = tag("pext");
              model.result().numerical().create(t, "EvalGlobal");
              model.result().numerical(t).set("data", "dset3");
              model.result().numerical(t).set("expr", expr);
              model.result().numerical(t).set("solnum", sol);
              double[][] re = model.result().numerical(t).getReal();
              double[][] im = model.result().numerical(t).getImag();
              double p0 = Math.hypot(re[idx0][0], im[idx0][0]);
              double spl0 = 20 * Math.log10(p0 / 20e-6);
              for (int k = 0; k < 181; k++) {
                int theta = -90 + k;
                double rad = Math.toRadians(theta);
                double rr = Math.abs(R * Math.sin(rad));
                double zz = R * Math.cos(rad);
                double pr = re[k][0], pi = im[k][0], pa = Math.hypot(pr, pi);
                double spl = 20 * Math.log10(pa / 20e-6);
                pw.println(String.join(",", csvq(v[2]), v[0], v[1], f(ft), f(fs[sol - 1]), Integer.toString(sol), f(R), Integer.toString(theta), f(rr), f(zz), f(pr), f(pi), f(pa), f(phase(pr, pi)), f(spl), f(spl - spl0), f(pa * R)));
              }
            } catch (Throwable t) {
              pw.println(csvq(v[2]) + "," + v[0] + "," + v[1] + "," + f(ft) + "," + f(fs[sol - 1]) + "," + sol + "," + f(R) + ",ERROR,,,,,,,,," + csvq(t.toString()));
            }
          }
        }
      }
    }
    }
    applyVariantToEfc1(model, "original");
  }

  static String applyVariantToEfc1(Model model, String label) {
    String status = "ok";
    try {
      setInt(model, "UsePPR", 1);
      setInt(model, "ReverseNormal", 0);
      setInt(model, "SymmetryCondition0", 0);
      setInt(model, "SymmetryCondition2", 1);
      try { model.component("comp1").physics("acpr").feature("efc1").set("IntType", "FullIntegral"); } catch (Throwable t) { status += ";reset_IntType_failed:" + t.getClass().getSimpleName(); }
      if ("no_symmetry".equals(label)) {
        setInt(model, "SymmetryCondition0", 0);
        setInt(model, "SymmetryCondition2", 0);
      } else if ("reverse_normal".equals(label)) {
        setInt(model, "ReverseNormal", 1);
      } else if ("farfield_approximation".equals(label)) {
        String[] candidates = {"FarFieldApproximation", "FarFieldIntegralApproximation", "FarFieldApprox", "FarField", "farfield"};
        boolean ok = false;
        String failures = "";
        for (String c : candidates) {
          try {
            model.component("comp1").physics("acpr").feature("efc1").set("IntType", c);
            ok = true;
            status += ";IntType=" + c;
            break;
          } catch (Throwable t) {
            failures += c + ":" + t.getClass().getSimpleName() + "|";
          }
        }
        if (!ok) status += ";all_farfield_IntType_candidates_failed:" + failures;
      }
    } catch (Throwable t) {
      status = "failed:" + t.toString();
    }
    return status;
  }

  static void setInt(Model model, String prop, int val) {
    try {
      model.component("comp1").physics("acpr").feature("efc1").set(prop, val);
    } catch (Throwable t) {
      model.component("comp1").physics("acpr").feature("efc1").setIndex(prop, val, 0);
    }
  }

  static void dumpEffective(PrintWriter pw, Model model, String label, String prop) {
    try {
      String v = model.component("comp1").physics("acpr").feature("efc1").getString(prop);
      pw.println(csvq(label) + "," + csvq(prop) + "," + csvq(v) + ",ok");
    } catch (Throwable t) {
      pw.println(csvq(label) + "," + csvq(prop) + ",," + csvq("failed:" + t.toString()));
    }
  }

  static void exportOriginalDirectivityPlot(Model model, String dir) throws Exception {
    try (PrintWriter pw = csv(dir, "req6_directivity_plot_export_status.csv", "export,status,filename")) {
      try {
        model.result("pg10").run();
        String ex = "req6_dir_plot";
        model.result().export().create(ex, "pg10", "dir1", "Plot");
        String fn = new File(dir, "req6_pg10_dir1_plot_export.csv").getPath();
        model.result().export(ex).set("filename", fn);
        model.result().export(ex).run();
        pw.println("pg10.dir1,ok,req6_pg10_dir1_plot_export.csv");
      } catch (Throwable t) {
        pw.println("pg10.dir1," + csvq("failed:" + t.toString()) + ",");
      }
    }
  }

  static void exportBoundary93GradientAudit(Model model, String dir) throws Exception {
    double[] fs = freqs(model, "dset3");
    String[] expr = {
      "r/1[m]", "z/1[m]", "nr", "nz", "acpr.p_t",
      "d(acpr.p_t,r)*nr+d(acpr.p_t,z)*nz",
      "ppr(d(acpr.p_t,r))*nr+ppr(d(acpr.p_t,z))*nz",
      "up(d(acpr.p_t,r))*nr+up(d(acpr.p_t,z))*nz",
      "down(d(acpr.p_t,r))*nr+down(d(acpr.p_t,z))*nz"
    };
    try (PrintWriter pw = csv(dir, "req6_boundary93_gradient_recovery_audit.csv",
      "requested_freq_Hz,solved_freq_Hz,solution_index,node_id,r_m,z_m,normal_r,normal_z,p_real,p_imag,dpdn_plain_real,dpdn_plain_imag,dpdn_ppr_real,dpdn_ppr_imag,dpdn_up_real,dpdn_up_imag,dpdn_down_real,dpdn_down_imag,status")) {
      for (double ft : FREQS) {
        int sol = nearestSol(fs, ft);
        try {
          String t = tag("b93");
          model.result().numerical().create(t, "Eval");
          model.result().numerical(t).set("data", "dset3");
          model.result().numerical(t).selection().geom("geom1", 1);
          model.result().numerical(t).selection().set(B93);
          model.result().numerical(t).set("expr", expr);
          model.result().numerical(t).set("solnum", sol);
          double[][] re = model.result().numerical(t).getReal();
          double[][] im = model.result().numerical(t).getImag();
          for (int i = 0; i < re.length; i++) {
            pw.println(String.join(",", f(ft), f(fs[sol - 1]), Integer.toString(sol), Integer.toString(i + 1),
              f(re[i][0]), f(re[i][1]), f(re[i][2]), f(re[i][3]), f(re[i][4]), f(im[i][4]),
              f(re[i][5]), f(im[i][5]), f(re[i][6]), f(im[i][6]), f(re[i][7]), f(im[i][7]), f(re[i][8]), f(im[i][8]), "ok"));
          }
        } catch (Throwable t) {
          pw.println(f(ft) + "," + f(fs[sol - 1]) + "," + sol + ",,,,,,,,,,,,,,,," + csvq("failed:" + t.toString()));
        }
      }
    }
  }

  static void writeReadme(String dir) throws Exception {
    try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(new File(dir, "README_REQ6_RAW.txt"), StandardCharsets.UTF_8)))) {
      pw.println("REQ6 raw export");
      pw.println("Purpose: audit COMSOL L9 Exterior Field Calculation operator settings and small pext/operator A-B comparisons.");
      pw.println("Key original EFC properties from probe/export: ExtName=pext, IntType=FullIntegral, UsePPR=1, ReverseNormal=0, SymmetryType=SymmetryPlanes, SymmetryCondition2=1.");
      pw.println("Directivity plot expression is acpr.efc1.Lp_pext with normalization=angle and normalizationangle=0.");
      pw.println("Angles are -90..90 deg, radii are 1,2,10 m, frequencies requested are 100,1000,6300 Hz.");
      pw.println("Axisymmetric pext evaluation uses r=abs(R*sin(theta)), z=R*cos(theta).");
    }
  }
}
