import java.nio.*;
import java.nio.file.*;
import java.util.*;
import yyy.ts.compress.camel.Camel;
import yyy.ts.compress.camel.CamelDecompressor;
public class CamelAudit {
 public static void main(String[] args) throws Exception {
  for(String path:args) {
   ByteBuffer b=ByteBuffer.wrap(Files.readAllBytes(Paths.get(path))).order(ByteOrder.LITTLE_ENDIAN);
   long n=(b.capacity()-16)/16, mismatches=0,bits=0; b.position((int)(16+n*8));
   double[] data=new double[(int)n];for(int i=0;i<n;i++)data[i]=b.getDouble();
   for(int start=0;start<n;start+=1000) {
    int end=(int)Math.min(start+1000,n);Camel c=new Camel();
    for(int i=start;i<end;i++)c.addValue(data[i]);c.close(); bits+=c.getSize();
    CamelDecompressor d=new CamelDecompressor(c.getOut());
    for(int i=start;i<end;i++)if(Double.doubleToRawLongBits(d.readPair())!=Double.doubleToRawLongBits(data[i]))mismatches++;
   }
   System.out.println(path+","+n+","+bits+","+mismatches);
  }
 }
}
