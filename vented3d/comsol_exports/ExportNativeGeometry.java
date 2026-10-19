import com.comsol.model.*;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.util.zip.GZIPOutputStream;

/** Extended node topology and coordinates of the stored native solution. */
public class ExportNativeGeometry {
  static File BASE;
  static void matrix(String name,double[][] a)throws Exception {
    try(DataOutputStream w=new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(new File(BASE,name+".f64.gz")))))) {
      w.writeInt(a.length);w.writeInt(a.length==0?0:a[0].length);
      for(double[] row:a)for(double x:row)w.writeDouble(x);
    }
  }
  static void matrix(String name,int[][] a)throws Exception {
    try(DataOutputStream w=new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(new File(BASE,name+".i32.gz")))))) {
      w.writeInt(a.length);w.writeInt(a.length==0?0:a[0].length);
      for(int[] row:a)for(int x:row)w.writeInt(x);
    }
  }
  public static void main(String[] args)throws Exception {
    if(args.length<2) throw new IllegalArgumentException("Usage: ExportNativeGeometry <solved.mph> <output-directory>");
    BASE=new File(args[1]);BASE.mkdirs();ModelUtil.initStandalone(false);
    Model m=ModelUtil.load("nativeGeometry",new File(args[0]).getCanonicalPath());
    try {
      XmeshInfo xi=m.sol("sol1").xmeshInfo();
      matrix("nodes_coords_model_unit",xi.nodes().coords());
      matrix("nodes_coords_geometry_unit",xi.nodes().gCoords());
      for(String type:new String[]{"tet","pyr","hex","tri","quad"}) {
        XmeshInfoElements el=xi.elements(type);
        matrix(type+"_nodes",el.nodes());matrix(type+"_local_coords",el.localCoords());
        matrix(type+"_dofs",el.dofs());matrix(type+"_local_dof_coords",el.localDofCoords());
        try(PrintWriter w=new PrintWriter(new File(BASE,type+"_dof_names.txt"))) {
          for(String name:el.localDofNames())w.println(name);
        }
        System.out.println("NATIVE_GEOMETRY "+type+" nodes="+el.nodes().length+" elements="+el.nodes()[0].length);
      }
    } finally {ModelUtil.remove("nativeGeometry");}
    try(PrintWriter w=new PrintWriter(new File(BASE,"EXPORT_COMPLETE.txt"))) {w.println("Read-only extended mesh export complete.");}
  }
}
