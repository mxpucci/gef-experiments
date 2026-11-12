import java.io.*; import java.nio.*; import java.nio.file.*;
import algorithms.DeXOR.encoder.ExactDeXOREncoder;
import algorithms.DeXOR.decoder.DoubleDeXORDecoder;
// args: in.f64 outprefix tmpdir -> outprefix.dexor_exact (int32 len + bytes per block); prints bytes and decode mismatches
public class AuditExact {
 public static void main(String[] a) throws Exception {
  ByteBuffer in=ByteBuffer.wrap(Files.readAllBytes(Paths.get(a[0]))).order(ByteOrder.LITTLE_ENDIAN);
  int nb=in.getInt(); long bytes=0, mis=0;
  try(DataOutputStream o=new DataOutputStream(new FileOutputStream(a[1]+".dexor_exact"))){
   for(int b=0;b<nb;b++){int n=in.getInt();double[] v=new double[n];for(int i=0;i<n;i++)v[i]=in.getDouble();
    String p=a[2]+"/x.bin"; new File(p).delete();
    ExactDeXOREncoder e=new ExactDeXOREncoder(p); for(double x:v) e.encode(x); e.flush();
    byte[] bs=Files.readAllBytes(Paths.get(p)); o.writeInt(bs.length); o.write(bs); bytes+=bs.length;
    DoubleDeXORDecoder d=new DoubleDeXORDecoder(p);
    for(int i=0;i<n;i++) if(Double.doubleToRawLongBits(d.decodeDouble())!=Double.doubleToRawLongBits(v[i])) mis++;
   }
  }
  System.out.println(a[0]+",exact_java_bytes="+bytes+",exact_java_mis="+mis);
 }
}
