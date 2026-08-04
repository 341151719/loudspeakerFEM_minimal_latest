import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class ComsolReq5Export {
  static final int B93 = 93;
  static final int MAX_BOUNDARY = 102;
  static final int[] ASB_BOUNDS = {15,16,17,19,21,23,25,27,31,32,34,39,40,42,43,44,46,47,48,49,51,56,57,61,62,63,64,66,67,68,69,74,75,77,78,79,80,81,91,92,99,100,101,102};
  static final int[] ACOUSTIC_DOMAINS = {1,3,5,8,9,10,11,12,13,14,15,16,20,21,22,25};
  static final int[] PML_DOMAINS = {1,5};
  static final int[] NRA_DOMAINS = {8,22};
  static final int[] SOLID_DOMAINS = {3,9,10,11,12,13,14,15,16,17,18,19,20,21,25};
  static final int[] COIL_DOMAINS = {17,18,19};
  static final int[] SOFT_IRON_DOMAINS = {6,23};
  static final int[] MAGNETIC_DOMAINS = {6,17,18,19,23,24};
  static final double[] CORE_FREQS = {20,50,100,600,630,1000,1300,2000,5000,6300,8000};
  static final double[] L2_FREQS = {50,900};
  static final double[] L7_FREQS = {600,630,6300};
  static int tagCounter = 0;

  public static void main(String[] args) throws Exception {
    try {
      run(args);
    } catch (Throwable t) {
      try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(new File("comsol_req5_exception.txt"), StandardCharsets.UTF_8)))) {
        t.printStackTrace(pw);
      }
      t.printStackTrace();
      throw t;
    }
  }

  static void run(String[] args) throws Exception {
    if (args == null) args = new String[0];
    Locale.setDefault(Locale.US);
    String mph = args.length > 0 ? args[0] : "loudspeaker_driver_req2_solved.mph";
    String outDir = args.length > 1 ? args[1] : "comsol_req5_raw";
    Files.createDirectories(Paths.get(outDir));
    ModelUtil.initStandalone(false);
    Model model = ModelUtil.load("req5", mph);

    exportModelInfo(model, outDir);
    exportL0Geometry(model, outDir);
    exportL1Magnetostatics(model, outDir);
    exportL2Induction(model, outDir);
    exportL3Impedance(model, outDir);
    exportL4Lorentz(model, outDir);
    exportL5Solid(model, outDir);
    exportL6Asb(model, outDir);
    exportL7Acoustic(model, outDir);
    exportL8Boundary93(model, outDir);
    exportL9Farfield(model, outDir);
    writeReadme(outDir);
    System.out.println("REQ5 raw export complete: " + new File(outDir).getAbsolutePath());
  }

  static PrintWriter csv(String dir, String name, String header) throws Exception {
    PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(new File(dir, name), StandardCharsets.UTF_8)));
    pw.println(header);
    return pw;
  }

  static String f(double v) {
    return (Double.isNaN(v) || Double.isInfinite(v)) ? "" : String.format(Locale.US, "%.15g", v);
  }

  static String q(String s) {
    if (s == null) return "";
    return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
  }

  static String csvq(String s) {
    if (s == null) return "";
    return s.contains(",") || s.contains("\"") ? "\"" + s.replace("\"", "\"\"") + "\"" : s;
  }

  static double phase(double re, double im) {
    return Math.atan2(im, re) * 180.0 / Math.PI;
  }

  static String tag(String prefix) {
    tagCounter++;
    return prefix + "_" + tagCounter;
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
    int best = 0;
    double err = Double.POSITIVE_INFINITY;
    for (int i = 0; i < fs.length; i++) {
      double e = Math.abs(fs[i] - target);
      if (e < err) { err = e; best = i; }
    }
    return best + 1;
  }

  static boolean isTarget(double f, double[] targets) {
    for (double t : targets) if (Math.abs(f - t) < 1e-6 * Math.max(1.0, t)) return true;
    return false;
  }

  static double[][][] evalBoundary(Model m, String ds, int boundary, int solnum, String[] expr) {
    String t = tag("b");
    m.result().numerical().create(t, "Eval");
    m.result().numerical(t).set("data", ds);
    m.result().numerical(t).selection().geom("geom1", 1);
    m.result().numerical(t).selection().set(boundary);
    m.result().numerical(t).set("expr", expr);
    m.result().numerical(t).set("solnum", solnum);
    return new double[][][]{m.result().numerical(t).getReal(), m.result().numerical(t).getImag()};
  }

  static double[][][] evalDomain(Model m, String ds, int domain, int solnum, String[] expr) {
    String t = tag("d");
    m.result().numerical().create(t, "Eval");
    m.result().numerical(t).set("data", ds);
    m.result().numerical(t).selection().set(domain);
    m.result().numerical(t).set("expr", expr);
    m.result().numerical(t).set("solnum", solnum);
    return new double[][][]{m.result().numerical(t).getReal(), m.result().numerical(t).getImag()};
  }

  static double[][][] intBoundary(Model m, String ds, int boundary, int solnum, String[] expr, boolean surface) {
    String t = tag("ib");
    m.result().numerical().create(t, "IntLine");
    m.result().numerical(t).set("data", ds);
    if (surface) m.result().numerical(t).set("intsurface", true);
    m.result().numerical(t).selection().geom("geom1", 1);
    m.result().numerical(t).selection().set(boundary);
    m.result().numerical(t).set("expr", expr);
    m.result().numerical(t).set("solnum", solnum);
    return new double[][][]{m.result().numerical(t).getReal(), m.result().numerical(t).getImag()};
  }

  static double[] arcLength(double[][] re) {
    double[] s = new double[re.length];
    for (int i = 1; i < re.length; i++) {
      double dr = re[i][0] - re[i - 1][0];
      double dz = re[i][1] - re[i - 1][1];
      s[i] = s[i - 1] + Math.hypot(dr, dz);
    }
    return s;
  }

  static double[] nodeWeights(double[] s) {
    double[] w = new double[s.length];
    if (s.length == 1) { w[0] = 0; return w; }
    for (int i = 0; i < s.length; i++) {
      if (i == 0) w[i] = 0.5 * (s[1] - s[0]);
      else if (i == s.length - 1) w[i] = 0.5 * (s[i] - s[i - 1]);
      else w[i] = 0.5 * (s[i + 1] - s[i - 1]);
    }
    return w;
  }

  static void exportModelInfo(Model m, String dir) throws Exception {
    try (PrintWriter pw = csv(dir, "layer00_model_inventory.csv", "kind,tag,label,notes")) {
      pw.println("comsol_version," + csvq(ModelUtil.getComsolVersion()) + ",,");
      for (String s : m.study().tags()) pw.println("study," + s + "," + csvq(m.study(s).label()) + ",");
      for (String d : m.result().dataset().tags()) pw.println("dataset," + d + "," + csvq(m.result().dataset(d).label()) + ",");
      for (String sel : m.component("comp1").selection().tags()) pw.println("selection," + sel + "," + csvq(m.component("comp1").selection(sel).label()) + ",");
      for (String mat : m.component("comp1").material().tags()) pw.println("material," + mat + "," + csvq(m.component("comp1").material(mat).label()) + ",");
      for (String ph : m.component("comp1").physics().tags()) pw.println("physics," + ph + "," + csvq(m.component("comp1").physics(ph).label()) + ",");
      for (String mp : m.component("comp1").multiphysics().tags()) pw.println("multiphysics," + mp + "," + csvq(m.component("comp1").multiphysics(mp).label()) + ",");
    }
  }

  static void exportL0Geometry(Model m, String dir) throws Exception {
    double[] fs = freqs(m, "dset3");
    int sol = nearestSol(fs, 6300);
    try (PrintWriter bp = csv(dir, "layer00_boundary_sample_points.csv",
      "boundary_id,node_id,s_arc_m,r_m,z_m,normal_r,normal_z,axisym_weight_2pi_r_ds")) {
      for (int b = 1; b <= MAX_BOUNDARY; b++) {
        try {
          double[][][] ri = evalBoundary(m, "dset3", b, sol, new String[]{"r/1[m]", "z/1[m]", "nr", "nz"});
          double[][] re = ri[0];
          double[] s = arcLength(re);
          double[] w = nodeWeights(s);
          for (int i = 0; i < re.length; i++) {
            double axisW = 2 * Math.PI * re[i][0] * w[i];
            bp.println(String.join(",", Integer.toString(b), Integer.toString(i + 1), f(s[i]), f(re[i][0]), f(re[i][1]), f(re[i][2]), f(re[i][3]), f(axisW)));
          }
        } catch (Throwable ignored) {}
      }
    }
    try (PrintWriter pw = csv(dir, "layer00_geometry_boundaries.csv",
      "boundary_id,n_nodes,r_mid_m,z_mid_m,length_m,axisymmetric_area_m2,normal_r_mean,normal_z_mean")) {
      for (int b = 1; b <= MAX_BOUNDARY; b++) {
        try {
          double[][][] ri = evalBoundary(m, "dset3", b, sol, new String[]{"r/1[m]", "z/1[m]", "nr", "nz"});
          double[][] re = ri[0];
          double[] s = arcLength(re);
          double[] w = nodeWeights(s);
          double area = 0, nr = 0, nz = 0;
          for (int i = 0; i < re.length; i++) {
            area += 2 * Math.PI * re[i][0] * w[i];
            nr += re[i][2];
            nz += re[i][3];
          }
          int mid = re.length / 2;
          pw.println(String.join(",", Integer.toString(b), Integer.toString(re.length), f(re[mid][0]), f(re[mid][1]), f(s.length == 0 ? 0 : s[s.length - 1]), f(area), f(nr / re.length), f(nz / re.length)));
        } catch (Throwable ignored) {}
      }
    }
    exportDomainSamples(m, dir, "dset3", "layer00_mesh_nodes.csv", allDomains(), new double[]{6300}, new String[]{"r/1[m]", "z/1[m]"}, 1000);
    try (PrintWriter pw = csv(dir, "layer00_mesh_elements.csv", "status,notes")) {
      pw.println("not_available_via_result_eval,Element connectivity is not exposed by the numerical Eval API used in this batch export; mesh node coordinates are exported in layer00_mesh_nodes.csv.");
    }
  }

  static int[] allDomains() {
    int[] d = new int[25];
    for (int i = 0; i < d.length; i++) d[i] = i + 1;
    return d;
  }

  static void exportL1Magnetostatics(Model m, String dir) throws Exception {
    exportDomainSamples(m, dir, "dset1", "layer01_magnetic_domain_points.csv", MAGNETIC_DOMAINS, new double[]{50,900},
      new String[]{"r/1[m]", "z/1[m]", "mf.Br", "mf.Bz", "mf.normB", "mf.normH", "mf.normB/(mu0_const*mf.normH)", "mf.Jiphi", "-mf.Br*N0*2*pi*r"}, 500);
    exportBoundarySamples(m, dir, "dset1", "layer01_magnetic_boundary_points.csv", ASB_BOUNDS, new double[]{50,900},
      new String[]{"r/1[m]", "z/1[m]", "nr", "nz", "mf.Br", "mf.Bz", "mf.normB", "mf.normH"}, 1000);
  }

  static void exportL2Induction(Model m, String dir) throws Exception {
    exportDomainSamples(m, dir, "dset1", "layer02_induction_current_points.csv", SOFT_IRON_DOMAINS, L2_FREQS,
      new String[]{"r/1[m]", "z/1[m]", "mf.Jiphi", "abs(mf.Jiphi)", "atan2(imag(mf.Jiphi),real(mf.Jiphi))*180/pi", "mf.Br", "mf.Bz", "mf.normB", "mf.normB/(mu0_const*mf.normH)"}, 1000);
    double[][][] ri = global(m, "dset1", new String[]{"freq", "mf.LCoil_1", "mf.RCoil_1", "mf.ZCoil_1", "mf.ICoil_1", "mf.PCoil_1"});
    try (PrintWriter pw = csv(dir, "layer02_blocked_impedance_full_sweep.csv",
      "freq_Hz,L_blocked_H,R_blocked_ohm,Z_blocked_real_ohm,Z_blocked_imag_ohm,Z_blocked_abs_ohm,I_blocked_real_A,I_blocked_imag_A,P_blocked_W")) {
      double[][] re = ri[0], im = ri[1];
      for (int i = 0; i < re[0].length; i++) {
        pw.println(String.join(",", f(re[0][i]), f(re[1][i]), f(re[2][i]), f(re[3][i]), f(im[3][i]), f(Math.hypot(re[3][i], im[3][i])), f(re[4][i]), f(im[4][i]), f(re[5][i])));
      }
    }
  }

  static void exportL3Impedance(Model m, String dir) throws Exception {
    double[][][] total = global(m, "dset3", new String[]{"freq", "mf.ICoil_1", "mf.ZCoil_1", "mf.PCoil_1", "mf.LCoil_1", "mf.RCoil_1", "pext(0,1[m])"});
    double[][][] blocked = global(m, "dset1", new String[]{"freq", "mf.ICoil_1", "mf.ZCoil_1", "mf.PCoil_1", "mf.LCoil_1", "mf.RCoil_1"});
    try (PrintWriter pw = csv(dir, "layer03_impedance_power_decomposition.csv",
      "freq_Hz,I_total_real_A,I_total_imag_A,Z_total_real_ohm,Z_total_imag_ohm,Z_total_abs_ohm,P_total_W,L_total_H,R_total_ohm,axis_pext_real_Pa,axis_pext_imag_Pa,I_blocked_real_A,I_blocked_imag_A,Z_blocked_real_ohm,Z_blocked_imag_ohm,Z_blocked_abs_ohm,P_blocked_W,L_blocked_H,R_blocked_ohm,Z_motional_real_ohm,Z_motional_imag_ohm")) {
      double[][] tr = total[0], ti = total[1], br = blocked[0], bi = blocked[1];
      int n = Math.min(tr[0].length, br[0].length);
      for (int i = 0; i < n; i++) {
        double zmr = tr[2][i] - br[2][i];
        double zmi = ti[2][i] - bi[2][i];
        pw.println(String.join(",", f(tr[0][i]), f(tr[1][i]), f(ti[1][i]), f(tr[2][i]), f(ti[2][i]), f(Math.hypot(tr[2][i], ti[2][i])), f(tr[3][i]), f(tr[4][i]), f(tr[5][i]), f(tr[6][i]), f(ti[6][i]), f(br[1][i]), f(bi[1][i]), f(br[2][i]), f(bi[2][i]), f(Math.hypot(br[2][i], bi[2][i])), f(br[3][i]), f(br[4][i]), f(br[5][i]), f(zmr), f(zmi)));
      }
    }
  }

  static void exportL4Lorentz(Model m, String dir) throws Exception {
    exportDomainSamples(m, dir, "dset3", "layer04_lorentz_force_density_points.csv", COIL_DOMAINS, CORE_FREQS,
      new String[]{"r/1[m]", "z/1[m]", "mf.Jiphi", "mf.Br", "mf.Bz", "mf.Jiphi*mf.Bz", "-mf.Jiphi*mf.Br", "u", "w"}, 1000);
  }

  static void exportL5Solid(Model m, String dir) throws Exception {
    exportDomainSamples(m, dir, "dset3", "layer05_solid_domain_points.csv", SOLID_DOMAINS, CORE_FREQS,
      new String[]{"r/1[m]", "z/1[m]", "u", "w", "solid.disp", "i*2*pi*freq*u", "i*2*pi*freq*w", "-(2*pi*freq)^2*u", "-(2*pi*freq)^2*w", "solid.Qh"}, 800);
    exportAsbPointLike(m, dir, "layer05_solid_asb_boundary_fields.csv", CORE_FREQS, false);
  }

  static void exportL6Asb(Model m, String dir) throws Exception {
    exportAsbPointLike(m, dir, "layer06_asb_coupling_point_fields.csv", CORE_FREQS, true);
    String[] surfExpr = {
      "freq",
      "real(acpr.p_t*conj(i*2*pi*freq*(u*nr+w*nz)))",
      "imag(acpr.p_t*conj(i*2*pi*freq*(u*nr+w*nz)))",
      "abs(acpr.p_t)^2",
      "abs(i*2*pi*freq*(u*nr+w*nz))^2",
      "abs(-(2*pi*freq)^2*(u*nr+w*nz))^2",
      "1",
      "abs(acpr.p_t)",
      "abs(i*2*pi*freq*(u*nr+w*nz))",
      "abs(-(2*pi*freq)^2*(u*nr+w*nz))"
    };
    try (PrintWriter pw = csv(dir, "layer06_asb_boundary_work_integrals.csv",
      "freq_Hz,boundary_id,int_2pi_r_Re_p_conj_vn,int_2pi_r_Im_p_conj_vn,int_2pi_r_abs_p2,int_2pi_r_abs_vn2,int_2pi_r_abs_an2,axisymmetric_area_m2,mean_abs_p,mean_abs_vn,mean_abs_an")) {
      double[] fs = freqs(m, "dset3");
      for (double target : CORE_FREQS) {
        int sol = nearestSol(fs, target);
        for (int b : ASB_BOUNDS) {
          try {
            double[][][] sr = intBoundary(m, "dset3", b, sol, surfExpr, true);
            double[][] re = sr[0];
            double area = re[6][0];
            double freq = area == 0 ? fs[sol - 1] : re[0][0] / area;
            pw.println(String.join(",", f(freq), Integer.toString(b), f(re[1][0]), f(re[2][0]), f(re[3][0]), f(re[4][0]), f(re[5][0]), f(area), f(area == 0 ? Double.NaN : re[7][0] / area), f(area == 0 ? Double.NaN : re[8][0] / area), f(area == 0 ? Double.NaN : re[9][0] / area)));
          } catch (Throwable ignored) {}
        }
      }
    }
  }

  static void exportL7Acoustic(Model m, String dir) throws Exception {
    exportDomainSamples(m, dir, "dset3", "layer07_acoustic_pressure_all_domains.csv", ACOUSTIC_DOMAINS, L7_FREQS,
      new String[]{"r/1[m]", "z/1[m]", "acpr.p_t", "abs(acpr.p_t)", "atan2(imag(acpr.p_t),real(acpr.p_t))*180/pi", "d(acpr.p_t,r)", "d(acpr.p_t,z)"}, 1000);
    exportDomainSamples(m, dir, "dset3", "layer07_acoustic_domain_points.csv", ACOUSTIC_DOMAINS, L7_FREQS,
      new String[]{"r/1[m]", "z/1[m]", "acpr.p_t", "abs(acpr.p_t)", "atan2(imag(acpr.p_t),real(acpr.p_t))*180/pi", "d(acpr.p_t,r)", "d(acpr.p_t,z)", "-d(acpr.p_t,r)/(i*2*pi*freq*acpr.rho)", "-d(acpr.p_t,z)/(i*2*pi*freq*acpr.rho)"}, 1000);
    exportDomainSamples(m, dir, "dset3", "layer07_nra_domain_points.csv", NRA_DOMAINS, new double[]{600,630},
      new String[]{"r/1[m]", "z/1[m]", "acpr.p_t", "abs(acpr.p_t)", "d(acpr.p_t,r)", "d(acpr.p_t,z)"}, 1000);
    exportDomainSamples(m, dir, "dset3", "layer07_pml_domain_points.csv", PML_DOMAINS, L7_FREQS,
      new String[]{"r/1[m]", "z/1[m]", "acpr.p_t", "abs(acpr.p_t)", "d(acpr.p_t,r)", "d(acpr.p_t,z)"}, 1000);
  }

  static void exportL8Boundary93(Model m, String dir) throws Exception {
    double[] fs = freqs(m, "dset3");
    String[] expr = {
      "r/1[m]", "z/1[m]", "nr", "nz", "acpr.p_t",
      "d(acpr.p_t,r)", "d(acpr.p_t,z)", "d(acpr.p_t,r)*nr+d(acpr.p_t,z)*nz",
      "-(d(acpr.p_t,r)*nr+d(acpr.p_t,z)*nz)/(i*2*pi*freq*acpr.rho)"
    };
    try (PrintWriter pw = csv(dir, "layer08_boundary93_source_points.csv",
      "freq_Hz,boundary_id,node_id,s_arc_m,r_m,z_m,normal_r,normal_z,ds_m,axisym_weight_2pi_r_ds,p_real_Pa,p_imag_Pa,p_abs_Pa,p_phase_deg,dp_dr_real,dp_dr_imag,dp_dz_real,dp_dz_imag,dp_dn_real,dp_dn_imag,v_n_real,v_n_imag,intensity_n_real_W_m2")) {
      for (double target : CORE_FREQS) {
        int sol = nearestSol(fs, target);
        double[][][] ri = evalBoundary(m, "dset3", B93, sol, expr);
        double[][] re = ri[0], im = ri[1];
        double[] s = arcLength(re);
        double[] w = nodeWeights(s);
        for (int i = 0; i < re.length; i++) {
          double pr = re[i][4], pi = im[i][4];
          double vnR = re[i][8], vnI = im[i][8];
          double intensity = 0.5 * (pr * vnR + pi * vnI);
          double axisW = 2 * Math.PI * re[i][0] * w[i];
          pw.println(String.join(",", f(fs[sol - 1]), Integer.toString(B93), Integer.toString(i + 1), f(s[i]), f(re[i][0]), f(re[i][1]), f(re[i][2]), f(re[i][3]), f(w[i]), f(axisW), f(pr), f(pi), f(Math.hypot(pr, pi)), f(phase(pr, pi)), f(re[i][5]), f(im[i][5]), f(re[i][6]), f(im[i][6]), f(re[i][7]), f(im[i][7]), f(vnR), f(vnI), f(intensity)));
        }
      }
    }
  }

  static void exportL9Farfield(Model m, String dir) throws Exception {
    String[] expr = new String[182];
    expr[0] = "freq";
    for (int i = 0; i <= 180; i++) {
      int theta = -90 + i;
      double rad = Math.toRadians(theta);
      double r = Math.abs(Math.sin(rad));
      double z = Math.cos(rad);
      expr[i + 1] = String.format(Locale.US, "pext(%.12g[m],%.12g[m])", r, z);
    }
    double[][][] ri = global(m, "dset3", expr);
    double[][] re = ri[0], im = ri[1];
    try (PrintWriter pw = csv(dir, "layer09_farfield_directivity_matrix.csv",
      "freq_Hz,theta_deg,pext_real_Pa,pext_imag_Pa,pext_abs_Pa,pext_phase_deg,SPL_abs_dB,SPL_relative_to_0deg_dB")) {
      for (double target : CORE_FREQS) {
        int j = nearestSol(re[0], target) - 1;
        double p0 = Math.hypot(re[91][j], im[91][j]);
        double spl0 = 20 * Math.log10(p0 / 20e-6);
        for (int i = 0; i <= 180; i++) {
          double pr = re[i + 1][j], pi = im[i + 1][j], pa = Math.hypot(pr, pi);
          double spl = 20 * Math.log10(pa / 20e-6);
          pw.println(String.join(",", f(re[0][j]), Integer.toString(-90 + i), f(pr), f(pi), f(pa), f(phase(pr, pi)), f(spl), f(spl - spl0)));
        }
      }
    }
  }

  static void exportAsbPointLike(Model m, String dir, String file, double[] targets, boolean withWork) throws Exception {
    double[] fs = freqs(m, "dset3");
    String header = withWork
      ? "freq_Hz,boundary_id,node_id,s_arc_m,r_m,z_m,normal_r,normal_z,p_real_Pa,p_imag_Pa,u_n_real,u_n_imag,v_n_real,v_n_imag,a_n_real,a_n_imag,pressure_work_density_real,pressure_work_density_imag"
      : "freq_Hz,boundary_id,node_id,s_arc_m,r_m,z_m,normal_r,normal_z,u_r_real,u_r_imag,u_z_real,u_z_imag,u_abs_m,v_r_real,v_r_imag,v_z_real,v_z_imag,a_r_real,a_r_imag,a_z_real,a_z_imag,u_n_real,u_n_imag,v_n_real,v_n_imag,a_n_real,a_n_imag";
    try (PrintWriter pw = csv(dir, file, header)) {
      String[] expr = {"r/1[m]", "z/1[m]", "nr", "nz", "acpr.p_t", "u", "w", "solid.disp"};
      for (double target : targets) {
        int sol = nearestSol(fs, target);
        double omega = 2 * Math.PI * fs[sol - 1];
        for (int b : ASB_BOUNDS) {
          try {
            double[][][] ri = evalBoundary(m, "dset3", b, sol, expr);
            double[][] re = ri[0], im = ri[1];
            double[] s = arcLength(re);
            for (int i = 0; i < re.length; i++) {
              double nr = re[i][2], nz = re[i][3];
              double pr = re[i][4], pi = im[i][4];
              double ur = re[i][5], ui = im[i][5], wzr = re[i][6], wzi = im[i][6];
              double vrR = -omega * ui, vrI = omega * ur;
              double vzR = -omega * wzi, vzI = omega * wzr;
              double arR = -omega * omega * ur, arI = -omega * omega * ui;
              double azR = -omega * omega * wzr, azI = -omega * omega * wzi;
              double unR = ur * nr + wzr * nz, unI = ui * nr + wzi * nz;
              double vnR = -omega * unI, vnI = omega * unR;
              double anR = -omega * omega * unR, anI = -omega * omega * unI;
              if (withWork) {
                double wrR = 0.5 * (pr * vnR + pi * vnI);
                double wrI = 0.5 * (pi * vnR - pr * vnI);
                pw.println(String.join(",", f(fs[sol - 1]), Integer.toString(b), Integer.toString(i + 1), f(s[i]), f(re[i][0]), f(re[i][1]), f(nr), f(nz), f(pr), f(pi), f(unR), f(unI), f(vnR), f(vnI), f(anR), f(anI), f(wrR), f(wrI)));
              } else {
                pw.println(String.join(",", f(fs[sol - 1]), Integer.toString(b), Integer.toString(i + 1), f(s[i]), f(re[i][0]), f(re[i][1]), f(nr), f(nz), f(ur), f(ui), f(wzr), f(wzi), f(re[i][7]), f(vrR), f(vrI), f(vzR), f(vzI), f(arR), f(arI), f(azR), f(azI), f(unR), f(unI), f(vnR), f(vnI), f(anR), f(anI)));
              }
            }
          } catch (Throwable ignored) {}
        }
      }
    }
  }

  static void exportDomainSamples(Model m, String dir, String ds, String file, int[] domains, double[] targets, String[] expr, int maxPerDomain) throws Exception {
    double[] fs = freqs(m, ds);
    StringBuilder header = new StringBuilder("freq_Hz,dataset_tag,solution_index,domain_id,node_id");
    for (String e : expr) header.append(',').append(e.replace(",", ";")).append("_real,").append(e.replace(",", ";")).append("_imag");
    try (PrintWriter pw = csv(dir, file, header.toString())) {
      for (double target : targets) {
        int sol = nearestSol(fs, target);
        for (int d : domains) {
          try {
            double[][][] ri = evalDomain(m, ds, d, sol, expr);
            double[][] re = ri[0], im = ri[1];
            int step = Math.max(1, re.length / maxPerDomain);
            int nid = 0;
            for (int i = 0; i < re.length; i += step) {
              nid++;
              StringBuilder sb = new StringBuilder();
              sb.append(f(fs[sol - 1])).append(',').append(ds).append(',').append(sol).append(',').append(d).append(',').append(nid);
              for (int k = 0; k < expr.length; k++) sb.append(',').append(f(re[i][k])).append(',').append(f(im[i][k]));
              pw.println(sb);
            }
          } catch (Throwable ignored) {}
        }
      }
    }
  }

  static void exportBoundarySamples(Model m, String dir, String ds, String file, int[] bounds, double[] targets, String[] expr, int maxPerBoundary) throws Exception {
    double[] fs = freqs(m, ds);
    StringBuilder header = new StringBuilder("freq_Hz,dataset_tag,solution_index,boundary_id,node_id");
    for (String e : expr) header.append(',').append(e.replace(",", ";")).append("_real,").append(e.replace(",", ";")).append("_imag");
    try (PrintWriter pw = csv(dir, file, header.toString())) {
      for (double target : targets) {
        int sol = nearestSol(fs, target);
        for (int b : bounds) {
          try {
            double[][][] ri = evalBoundary(m, ds, b, sol, expr);
            double[][] re = ri[0], im = ri[1];
            int step = Math.max(1, re.length / maxPerBoundary);
            int nid = 0;
            for (int i = 0; i < re.length; i += step) {
              nid++;
              StringBuilder sb = new StringBuilder();
              sb.append(f(fs[sol - 1])).append(',').append(ds).append(',').append(sol).append(',').append(b).append(',').append(nid);
              for (int k = 0; k < expr.length; k++) sb.append(',').append(f(re[i][k])).append(',').append(f(im[i][k]));
              pw.println(sb);
            }
          } catch (Throwable ignored) {}
        }
      }
    }
  }

  static void writeReadme(String dir) throws Exception {
    String text = "REQ5 raw export\n"
      + "Purpose: COMSOL intermediate-layer black-box decomposition for L0-L9 contracts.\n"
      + "Core diagnostic frequency set: 20,50,100,600,630,1000,1300,2000,5000,6300,8000 Hz.\n"
      + "Priority coverage: 6300 Hz L5 solid, L6 ASB, L8 Boundary93, L9 far-field; 600/630 Hz NRA/acoustic; 50/900 Hz induction; full-sweep impedance decomposition.\n"
      + "Mesh connectivity is noted as unavailable through this result-evaluation batch path; mesh node coordinates and boundary sample weights are exported.\n";
    Files.write(Paths.get(dir, "README_REQ5_RAW.txt"), text.getBytes(StandardCharsets.UTF_8));
  }
}
