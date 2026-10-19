import com.comsol.model.*;
import com.comsol.model.util.ModelUtil;
import java.io.*;
import java.util.zip.GZIPOutputStream;

/** Exact stored SI solution vectors. No study run, no model save. */
public class ExportStoredVectors {
  public static void main(String[] args)throws Exception {
    if(args.length<2) throw new IllegalArgumentException("Usage: ExportStoredVectors <solved.mph> <output-directory>");
    File out=new File(args[1]);out.mkdirs();
    ModelUtil.initStandalone(false);
    Model m=ModelUtil.load("vectors",new File(args[0]).getCanonicalPath());
    try {
      for(int index:new int[]{5,7,25,46}) {
        double[] re=m.sol("sol1").getU(index),im=m.sol("sol1").getUImag(index);
        if(re.length!=im.length)throw new IllegalStateException("Vector length mismatch");
        try(DataOutputStream w=new DataOutputStream(new BufferedOutputStream(new GZIPOutputStream(new FileOutputStream(new File(out,"sol1_"+index+".bin.gz")))))) {
          w.writeInt(1);w.writeInt(re.length);
          for(int i=0;i<re.length;i++) {
            if(!Double.isFinite(re[i])||!Double.isFinite(im[i]))throw new IllegalStateException("Nonfinite vector");
            w.writeDouble(re[i]);w.writeDouble(im[i]);
          }
        }
        System.out.println("STORED_VECTOR_OK index="+index+" length="+re.length);
      }
    } finally {ModelUtil.remove("vectors");}
    try(PrintWriter w=new PrintWriter(new File(out,"EXPORT_COMPLETE.txt"))) {w.println("Stored solutions 5,7,25,46 exported; no study execution.");}
  }
}
