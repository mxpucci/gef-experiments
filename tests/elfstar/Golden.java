import org.urbcomp.startdb.selfstar.compressor.ElfStarCompressor;
import org.urbcomp.startdb.selfstar.compressor.xor.ElfStarXORCompressor;
public class Golden {
 public static void main(String[] args) {
  double[] values={1.1,2.2,3.3,3.3,3.3,-4.4,0.1,0.01,0.001,123.456,18.18,18.18,0.0,-0.0,42.0,42.000001};
  ElfStarCompressor c=new ElfStarCompressor(new ElfStarXORCompressor());
  for(double v:values)c.addValue(v);c.close();
  System.out.println(c.getCompressedSizeInBits());
  for(byte b:c.getBytes())System.out.printf("%02x",b&255);
  System.out.println();
 }
}
