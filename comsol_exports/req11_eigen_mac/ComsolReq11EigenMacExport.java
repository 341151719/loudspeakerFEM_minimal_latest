import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/** Export all ten dset7 structural eigenvectors for mass-MAC pairing. */
public class ComsolReq11EigenMacExport {
  static final int[] DOMAINS={3,9,10,11,12,13,14,15,16,17,18,19,20,21,25};
  static String f(double x){return String.format(Locale.US,"%.15g",x);}
  public static void main(String[] args)throws Exception{
    Locale.setDefault(Locale.US);
    String mph=args.length>0?args[0]:"loudspeaker_driver_solved.mph";
    String out=args.length>1?args[1]:"comsol_req11_eigen_mac_raw";
    Files.createDirectories(Paths.get(out));ModelUtil.initStandalone(false);Model m=ModelUtil.load("req11",mph);
    String ds="dset7";
    String gt="req11_freq";m.result().numerical().create(gt,"EvalGlobal");m.result().numerical(gt).set("data",ds);m.result().numerical(gt).set("expr",new String[]{"freq"});
    double[][] fr=m.result().numerical(gt).getReal(),fi=m.result().numerical(gt).getImag();int nm=fr[0].length;
    try(PrintWriter p=new PrintWriter(new BufferedWriter(new FileWriter(new File(out,"eigenfrequencies.csv"),StandardCharsets.UTF_8)))){
      p.println("mode_index,eigenfrequency_real_Hz,eigenfrequency_imag_Hz");for(int k=0;k<nm;k++)p.println((k+1)+","+f(fr[0][k])+","+f(fi[0][k]));
    }
    try(PrintWriter p=new PrintWriter(new BufferedWriter(new FileWriter(new File(out,"eigenmode_shapes_all.csv"),StandardCharsets.UTF_8)))){
      p.println("mode_index,freq_Hz,domain_id,node_id,r_m,z_m,u_real,u_imag,w_real,w_imag,solid_disp_real,solid_disp_imag");
      String[] ex={"r/1[m]","z/1[m]","u","w","solid.disp"};
      for(int k=1;k<=nm;k++)for(int dom:DOMAINS){
        String t="req11_m"+k+"_d"+dom;m.result().numerical().create(t,"Eval");m.result().numerical(t).set("data",ds);m.result().numerical(t).selection().geom("geom1",2);m.result().numerical(t).selection().set(dom);m.result().numerical(t).set("expr",ex);m.result().numerical(t).set("solnum",k);m.result().numerical(t).set("complexfun","on");
        double[][] re=m.result().numerical(t).getReal(),im=m.result().numerical(t).getImag();
        for(int i=0;i<re.length;i++)p.println(k+","+f(fr[0][k-1])+","+dom+","+(i+1)+","+f(re[i][0])+","+f(re[i][1])+","+f(re[i][2])+","+f(im[i][2])+","+f(re[i][3])+","+f(im[i][3])+","+f(re[i][4])+","+f(im[i][4]));
      }
    }
    Files.writeString(Paths.get(out,"README.txt"),"All dset7 eigenvectors exported by explicit solnum=1..N. This fixes the previous exporter bug that interpreted mode indices 1,2,3,4 as target frequencies and therefore repeatedly selected mode 1.\n");
    System.out.println("REQ11 complete: "+new File(out).getAbsolutePath());
  }
}
