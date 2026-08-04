import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * REQ8: export the actual nonzero small-signal Magnetomechanics force.
 *
 * Run with COMSOL's javac/java wrappers against the solved MPH. The nonlinear
 * Lorentz-force variable is linearized explicitly with lindev(...), equivalent
 * to Results > Compute differential on the force expression.
 */
public class ComsolReq8L4Export {
  static final int[] COIL_DOMAINS={17,18,19};
  static int counter=0;
  static String tag(String p){return p+(++counter);}
  static String f(double x){return Double.isFinite(x)?String.format(Locale.US,"%.15g",x):"";}
  static String q(String s){return "\""+s.replace("\"","'")+"\"";}
  static PrintWriter csv(String dir,String name,String header)throws Exception{
    PrintWriter p=new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(dir,name)),StandardCharsets.UTF_8)));p.println(header);return p;
  }
  public static void main(String[] args)throws Exception{
    Locale.setDefault(Locale.US);
    String mph=args.length>0?args[0]:"loudspeaker_driver_solved.mph";
    String out=args.length>1?args[1]:"comsol_req8_l4_raw";
    Files.createDirectories(Paths.get(out));ModelUtil.initStandalone(false);Model m=ModelUtil.load("req8",mph);
    double[] fs=freqs(m,"dset3");String[] chosen=probeAndChoose(m,out,fs);
    exportProperties(m,out,chosen);exportPoints(m,out,fs,chosen[0],chosen[1]);exportIntegrals(m,out,fs,chosen[0],chosen[1]);
    Files.write(Paths.get(out,"README_REQ8_L4_RAW.txt"),("REQ8 actual Magnetomechanics L4 export\nSelected variables: "+chosen[0]+", "+chosen[1]+"\nForce export operator: lindev\n").getBytes(StandardCharsets.UTF_8));
    System.out.println("REQ8 L4 export complete: "+new File(out).getAbsolutePath());
  }
  static void configure(Model m,String t,String ds,int solnum,String[] expr,boolean differential){
    m.result().numerical(t).set("data",ds);m.result().numerical(t).set("expr",expr);m.result().numerical(t).set("solnum",solnum);
    m.result().numerical(t).set("evalmethod","harmonic");m.result().numerical(t).set("differential",differential?"on":"off");m.result().numerical(t).set("complexfun","on");m.result().numerical(t).set("matherr","on");
  }
  static double[][][] evalDomain(Model m,String ds,int[] domains,int solnum,String[] expr,boolean differential){
    String t=tag("ed");m.result().numerical().create(t,"Eval");m.result().numerical(t).selection().geom("geom1",2);m.result().numerical(t).selection().set(domains);configure(m,t,ds,solnum,expr,differential);return new double[][][]{m.result().numerical(t).getReal(),m.result().numerical(t).getImag()};
  }
  static double[][][] integrateVolume(Model m,String ds,int[] domains,int solnum,String[] expr){
    String t=tag("iv");m.result().numerical().create(t,"IntSurface");m.result().numerical(t).selection().geom("geom1",2);m.result().numerical(t).selection().set(domains);configure(m,t,ds,solnum,expr,false);m.result().numerical(t).set("intvolume",true);m.result().numerical(t).set("intorderactive","on");m.result().numerical(t).set("intorder",6);return new double[][][]{m.result().numerical(t).getReal(),m.result().numerical(t).getImag()};
  }
  static double[][][] global(Model m,String ds,int solnum,String[] expr){
    String t=tag("g");m.result().numerical().create(t,"EvalGlobal");configure(m,t,ds,solnum,expr,false);return new double[][][]{m.result().numerical(t).getReal(),m.result().numerical(t).getImag()};
  }
  static double[] freqs(Model m,String ds){String t=tag("gf");m.result().numerical().create(t,"EvalGlobal");m.result().numerical(t).set("data",ds);m.result().numerical(t).set("expr",new String[]{"freq"});return m.result().numerical(t).getReal()[0];}
  static int nearest(double[] fs,double target){int b=0;double e=Double.POSITIVE_INFINITY;for(int i=0;i<fs.length;i++){double x=Math.abs(fs[i]-target);if(x<e){e=x;b=i;}}return b+1;}
  static double maxAbs(double[][] re,double[][] im,int col){double z=0;for(int i=0;i<re.length;i++)z=Math.max(z,Math.hypot(re[i][col],im[i][col]));return z;}
  static String[] probeAndChoose(Model m,String out,double[] fs)throws Exception{
    String[] radial={"mmcpl1.FLtzr","mf.FLtzr","mmcpl1.FLtr","mf.FLtr"};String[] axial={"mmcpl1.FLtzz","mf.FLtzz","mmcpl1.FLtz","mf.FLtz"};
    ArrayList<String> all=new ArrayList<>();all.addAll(Arrays.asList(radial));all.addAll(Arrays.asList(axial));all.addAll(Arrays.asList(new String[]{"mmcpl1.FLtzavr","mmcpl1.FLtzavz","mf.Jphi","mf.Jiphi","mf.Br","mf.Bz"}));
    int sol=nearest(fs,500);String bestR=null,bestZ=null;double bestRv=0,bestZv=0;
    try(PrintWriter pw=csv(out,"l4_expression_probe.csv","expression,evaluation,status,n_points,max_abs,message")){
      for(String ex:all){try{double[][][] x=evalDomain(m,"dset3",COIL_DOMAINS,sol,new String[]{"lindev("+ex+")"},false);double ma=maxAbs(x[0],x[1],0);pw.println(ex+",explicit_lindev,ok,"+x[0].length+","+f(ma)+",");if(ma>bestRv&&Arrays.asList(radial).contains(ex)){bestRv=ma;bestR=ex;}if(ma>bestZv&&Arrays.asList(axial).contains(ex)){bestZv=ma;bestZ=ex;}}catch(Throwable t){pw.println(ex+",explicit_lindev,failed,0,,"+q(t.toString()));}}
    }
    if(bestR==null||bestZ==null)throw new RuntimeException("No nonzero differential Magnetomechanics force pair found; inspect l4_expression_probe.csv and Equation View.");
    try(PrintWriter pw=csv(out,"l4_selected_expressions.csv","role,base_expression,export_expression,max_abs_probe")){pw.println("radial_force_density,"+bestR+",lindev("+bestR+"),"+f(bestRv));pw.println("axial_force_density,"+bestZ+",lindev("+bestZ+"),"+f(bestZv));}
    return new String[]{bestR,bestZ};
  }
  static void exportProperties(Model m,String out,String[] chosen)throws Exception{
    try(PrintWriter pw=csv(out,"l4_feature_properties.csv","key,value")){pw.println("comsol_version,"+ModelUtil.getComsolVersion());pw.println("dataset,dset3");pw.println("multiphysics_tag,mmcpl1");pw.println("coil_domains,17;18;19");pw.println("force_r_base_expression,"+chosen[0]);pw.println("force_z_base_expression,"+chosen[1]);pw.println("force_export_operator,lindev");pw.println("axisymmetric_volume_integration,intvolume=true");pw.println("integration_order,6");}
  }
  static void exportPoints(Model m,String out,double[] fs,String fr,String fz)throws Exception{
    String[] expr={"r/1[m]","z/1[m]","lindev("+fr+")","lindev("+fz+")","linper(mf.Jphi)","linper(mf.Jiphi)","linper(mf.Br)","linper(mf.Bz)","u","w","i*2*pi*freq*u","i*2*pi*freq*w"};
    String header="freq_Hz,solution_index,domain_id,node_id,r_m,z_m,force_r_real_N_per_m3,force_r_imag_N_per_m3,force_z_real_N_per_m3,force_z_imag_N_per_m3,Jphi_real_A_per_m2,Jphi_imag_A_per_m2,Jiphi_real_A_per_m2,Jiphi_imag_A_per_m2,Br_real_T,Br_imag_T,Bz_real_T,Bz_imag_T,u_real_m,u_imag_m,w_real_m,w_imag_m,vr_real_m_per_s,vr_imag_m_per_s,vz_real_m_per_s,vz_imag_m_per_s";
    try(PrintWriter pw=csv(out,"l4_magnetomechanics_force_points.csv",header)){for(int si=1;si<=fs.length;si++)for(int dom:COIL_DOMAINS){double[][][] x=evalDomain(m,"dset3",new int[]{dom},si,expr,false);double[][] re=x[0],im=x[1];int step=Math.max(1,re.length/800),nid=0;for(int i=0;i<re.length;i+=step){nid++;StringBuilder s=new StringBuilder();s.append(f(fs[si-1])).append(',').append(si).append(',').append(dom).append(',').append(nid);for(int k=0;k<expr.length;k++)s.append(',').append(f(re[i][k])).append(',').append(f(im[i][k]));pw.println(s);}}}
  }
  static void exportIntegrals(Model m,String out,double[] fs,String fr,String fz)throws Exception{
    String frd="lindev("+fr+")",fzd="lindev("+fz+")",vr="i*2*pi*freq*u",vz="i*2*pi*freq*w";
    String[] ex={frd,fzd,"abs("+frd+")^2+abs("+fzd+")^2","r*"+fzd,"z*"+frd,frd+"*conj("+vr+")+"+fzd+"*conj("+vz+")","1"};
    try(PrintWriter pw=csv(out,"l4_magnetomechanics_force_integrals.csv","freq_Hz,solution_index,domain_id,force_r_real_N,force_r_imag_N,force_z_real_N,force_z_imag_N,force_L2_N_per_sqrt_m3,moment_rFz_real_Nm,moment_rFz_imag_Nm,moment_zFr_real_Nm,moment_zFr_imag_Nm,force_conj_velocity_real,force_conj_velocity_imag,axisymmetric_volume_m3,I_real_A,I_imag_A,force_z_per_I_real_N_per_A,force_z_per_I_imag_N_per_A,Ztotal_real_ohm,Ztotal_imag_ohm")){
      for(int si=1;si<=fs.length;si++){double[][][] gg=global(m,"dset3",si,new String[]{"mf.ICoil_1","mf.ZCoil_1"});double Ir=gg[0][0][0],Ii=gg[1][0][0],ztR=gg[0][1][0],ztI=gg[1][1][0];for(int di=-1;di<COIL_DOMAINS.length;di++){int[] doms=di<0?COIL_DOMAINS:new int[]{COIL_DOMAINS[di]};int label=di<0?0:COIL_DOMAINS[di];double[][][] a=integrateVolume(m,"dset3",doms,si,ex);double[][] r=a[0],im=a[1];double FrR=r[0][0],FrI=im[0][0],FzR=r[1][0],FzI=im[1][0],den=Ir*Ir+Ii*Ii,qR=(FzR*Ir+FzI*Ii)/den,qI=(FzI*Ir-FzR*Ii)/den;pw.println(String.join(",",f(fs[si-1]),Integer.toString(si),Integer.toString(label),f(FrR),f(FrI),f(FzR),f(FzI),f(Math.sqrt(Math.max(0,r[2][0]))),f(r[3][0]),f(im[3][0]),f(r[4][0]),f(im[4][0]),f(r[5][0]),f(im[5][0]),f(r[6][0]),f(Ir),f(Ii),f(qR),f(qI),f(ztR),f(ztI)));}}
    }
  }
}
