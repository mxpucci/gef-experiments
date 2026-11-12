import java.io.*;
import java.util.Arrays;
import yyy.ts.compress.camel.Camel;
import yyy.ts.compress.camel.CamelDecompressor;

// Bitstream parity harness for the native port (tests/camel_dump.cpp).
// Input: little-endian u64 count, then IEEE doubles. Encodes independent
// 1000-value blocks (final partial block kept) with the authors' Camel and
// writes, per block, a little-endian u32 byte length followed by the bytes
// (OutputBitStream.writtenBits() after close(), rounded up to bytes), plus
// the doubles decoded by the authors' CamelDecompressor.
public class CamelDump {
 public static void main(String[] args) throws IOException {
  double[] data;
  try(DataInputStream in=new DataInputStream(new BufferedInputStream(new FileInputStream(args[0])))) {
   int n=Math.toIntExact(Long.reverseBytes(in.readLong()));data=new double[n];
   for(int i=0;i<n;i++)data[i]=Double.longBitsToDouble(Long.reverseBytes(in.readLong()));
  }
  final int block=1000;
  try(DataOutputStream blocks=new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[1])));
      DataOutputStream decoded=new DataOutputStream(new BufferedOutputStream(new FileOutputStream(args[2])))) {
   for(int start=0;start<data.length;start+=block) {
    int end=Math.min(start+block,data.length);
    Camel c=new Camel();for(int i=start;i<end;i++)c.addValue(data[i]);c.close();
    byte[] b=Arrays.copyOf(c.getOut(),Math.toIntExact((c.getOutputStream().writtenBits()+7)/8));
    blocks.writeInt(Integer.reverseBytes(b.length));blocks.write(b);
    CamelDecompressor d=new CamelDecompressor(b);
    for(int i=start;i<end;i++)decoded.writeLong(Long.reverseBytes(Double.doubleToRawLongBits(d.readPair())));
   }
  }
 }
}
