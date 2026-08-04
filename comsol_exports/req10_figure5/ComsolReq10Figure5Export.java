import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class ComsolReq10Figure5Export {
  static final int[] DOMAINS = {6,23};
  static final double[] FREQS = {50,900,2000,5000,8000};
  static int count=0;
  static String tag(String p){return p+"_"+(++count);}  
  static String f(double x){return (Double.isNaN(x)||Double.isInfinite(x))?"":String.format(Locale.US,"%.15g",x);}  
  static String q(String s){return "\""+s.replace("\"","\"\"")+"\"";}  
  static PrintWriter csv(String dir,String name,String header)throws Exception{PrintWriter p=new PrintWriter(new BufferedWriter(new FileWriter(new File(dir,name),StandardCharsets.UTF_8)));p.println(header);return p;}

  public static void main(String[] args)throws Exception{
    Locale.setDefault(Locale.US);
    String mph=args.length>0?args[0]:"loudspeaker_driver_req2_solved.mph";
    String out=args.length>1?args[1]:"comsol_req10_figure5_raw";
    Files.createDirectories(Paths.get(out));ModelUtil.initStandalone(false);Model m=ModelUtil.load("req10",mph);
    exportInventory(m,out);exportExpressionProbe(m,out);exportPointFields(m,out);exportDomainLosses(m,out);exportGlobalBlocked(m,out);writeReadme(out);
    System.out.println("REQ10 Figure5 export complete: "+new File(out).getAbsolutePath());
  }
  static void configure(Model m,String t,String ds,int sol,String[] expr){m.result().numerical(t).set("data",ds);m.result().numerical(t).set("expr",expr);m.result().numerical(t).set("solnum",sol);}
  static double[][][] evalDomain(Model m,String ds,int dom,int sol,String[] expr){String t=tag("ed");m.result().numerical().create(t,"Eval");m.result().numerical(t).selection().geom("geom1",2);m.result().numerical(t).selection().set(dom);configure(m,t,ds,sol,expr);return new double[][][]{m.result().numerical(t).getReal(),m.result().numerical(t).getImag()};}
  static double[][][] intDomain(Model m,String ds,int dom,int sol,String[] expr){String t=tag("iv");m.result().numerical().create(t,"IntSurface");m.result().numerical(t).selection().geom("geom1",2);m.result().numerical(t).selection().set(dom);configure(m,t,ds,sol,expr);m.result().numerical(t).set("intvolume",true);m.result().numerical(t).set("intorderactive","on");m.result().numerical(t).set("intorder",8);return new double[][][]{m.result().numerical(t).getReal(),m.result().numerical(t).getImag()};}
  static double[][][] global(Model m,String ds,int sol,String[] expr){String t=tag("g");m.result().numerical().create(t,"EvalGlobal");configure(m,t,ds,sol,expr);return new double[][][]{m.result().numerical(t).getReal(),m.result().numerical(t).getImag()};}
  static double[] freqs(Model m){String t=tag("gf");m.result().numerical().create(t,"EvalGlobal");m.result().numerical(t).set("data","dset1");m.result().numerical(t).set("expr",new String[]{"freq"});return m.result().numerical(t).getReal()[0];}
  static int nearest(double[] fs,double target){int b=0;double e=Double.POSITIVE_INFINITY;for(int i=0;i<fs.length;i++){double x=Math.abs(fs[i]-target);if(x<e){e=x;b=i;}}return b+1;}

  static void exportInventory(Model m,String out)throws Exception{
    try(PrintWriter p=csv(out,"figure5_export_inventory.csv","key,value")){p.println("comsol_version,"+q(ModelUtil.getComsolVersion()));p.println("dataset,dset1");p.println("domains,6;23");p.println("frequencies_Hz,50;900;2000;5000;8000");p.println("phasor_loss_convention,0.5*abs(J)^2/sigma");p.println("nominal_soft_iron_sigma_S_per_m,1.12e7");p.println("axisymmetric_volume_integration,intvolume=true");p.println("integration_order,8");}
  }
  static void exportExpressionProbe(Model m,String out)throws Exception{
    double[] fs=freqs(m);int sol=nearest(fs,2000);String[] ex={"mf.Jiphi","mf.Jphi","mf.Qh","mf.Qrh","mf.normJ","mf.Br","mf.Bz"};
    try(PrintWriter p=csv(out,"figure5_expression_probe.csv","expression,status,n_points,max_abs,message")){
      for(String e:ex){try{double[][][] x=evalDomain(m,"dset1",6,sol,new String[]{e});double z=0;for(int i=0;i<x[0].length;i++)z=Math.max(z,Math.hypot(x[0][i][0],x[1][i][0]));p.println(e+",ok,"+x[0].length+","+f(z)+",");}catch(Throwable t){p.println(e+",failed,0,,"+q(t.toString()));}}
    }
  }
  static void exportPointFields(Model m,String out)throws Exception{
    double[] fs=freqs(m);String[] ex={"r/1[m]","z/1[m]","mf.Jiphi","mf.Jphi","mf.Br","mf.Bz","mf.normB","mf.normH","mf.normB/(mu0_const*mf.normH)"};
    String h="requested_freq_Hz,solved_freq_Hz,solution_index,domain_id,node_id,r_m,z_m,Jiphi_real_A_m2,Jiphi_imag_A_m2,Jphi_real_A_m2,Jphi_imag_A_m2,Br_real_T,Br_imag_T,Bz_real_T,Bz_imag_T,Bnorm_T,Hnorm_A_m,mu_effective_relative";
    try(PrintWriter p=csv(out,"figure5_Jphi_domain_points.csv",h)){
      for(double ft:FREQS){int sol=nearest(fs,ft);if(Math.abs(fs[sol-1]-ft)>1e-6*Math.max(1,ft))throw new RuntimeException("requested frequency not solved: "+ft+" nearest="+fs[sol-1]);for(int dom:DOMAINS){double[][][] x=evalDomain(m,"dset1",dom,sol,ex);double[][] r=x[0],im=x[1];for(int i=0;i<r.length;i++){p.println(String.join(",",f(ft),f(fs[sol-1]),Integer.toString(sol),Integer.toString(dom),Integer.toString(i+1),f(r[i][0]),f(r[i][1]),f(r[i][2]),f(im[i][2]),f(r[i][3]),f(im[i][3]),f(r[i][4]),f(im[i][4]),f(r[i][5]),f(im[i][5]),f(r[i][6]),f(r[i][7]),f(r[i][8])));}}}
    }
  }
  static void exportDomainLosses(Model m,String out)throws Exception{
    double[] fs=freqs(m);String[] ex={"abs(mf.Jiphi)^2/(2*1.12e7[S/m])","abs(mf.Jphi)^2/(2*1.12e7[S/m])","abs(mf.Jiphi)^2","1"};
    try(PrintWriter p=csv(out,"figure5_domain_joule_loss.csv","requested_freq_Hz,solved_freq_Hz,solution_index,domain_id,Jiphi_loss_W,Jphi_loss_W,Jiphi2_integral_A2_m_minus1,axisymmetric_volume_m3,mf_Qh_W_optional,mf_Qrh_W_optional")){
      for(double ft:FREQS){int sol=nearest(fs,ft);for(int dom:DOMAINS){double[][][] x=intDomain(m,"dset1",dom,sol,ex);String qh="",qrh="";try{double[][][] z=intDomain(m,"dset1",dom,sol,new String[]{"mf.Qh"});qh=f(z[0][0][0]);}catch(Throwable ignored){}try{double[][][] z=intDomain(m,"dset1",dom,sol,new String[]{"mf.Qrh"});qrh=f(z[0][0][0]);}catch(Throwable ignored){}p.println(String.join(",",f(ft),f(fs[sol-1]),Integer.toString(sol),Integer.toString(dom),f(x[0][0][0]),f(x[0][1][0]),f(x[0][2][0]),f(x[0][3][0]),qh,qrh));}}
    }
  }
  static void exportGlobalBlocked(Model m,String out)throws Exception{
    double[] fs=freqs(m);try(PrintWriter p=csv(out,"figure5_blocked_global.csv","requested_freq_Hz,solved_freq_Hz,solution_index,I_real_A,I_imag_A,Z_real_ohm,Z_imag_ohm,R_ohm,L_H,P_W")){
      for(double ft:FREQS){int sol=nearest(fs,ft);double[][][] x=global(m,"dset1",sol,new String[]{"mf.ICoil_1","mf.ZCoil_1","mf.RCoil_1","mf.LCoil_1","mf.PCoil_1"});p.println(String.join(",",f(ft),f(fs[sol-1]),Integer.toString(sol),f(x[0][0][0]),f(x[1][0][0]),f(x[0][1][0]),f(x[1][1][0]),f(x[0][2][0]),f(x[0][3][0]),f(x[0][4][0])));}
    }
  }
  static void writeReadme(String out)throws Exception{String s="REQ10 Figure 5 direct field export\nDataset: dset1 blocked induction-current study.\nDomains: 6 and 23. Frequencies: 50, 900, 2000, 5000, 8000 Hz.\nNo point subsampling is applied. Joule loss is exported independently per domain using axisymmetric volume integration and 0.5*|J|^2/sigma for peak phasors.\n";Files.write(Paths.get(out,"README_REQ10_FIGURE5_RAW.txt"),s.getBytes(StandardCharsets.UTF_8));}
}
