import algorithms.DeXOR.encoder.DoubleDeXOREncoder;
import algorithms.DeXOR.decoder.DoubleDeXORDecoder;
import java.nio.file.*;
// Prints hex of the stock Java DeXOR bitstream and the stock Java decoder's output bits.
public class Fixture { public static void main(String[] a) throws Exception {
 double[] v={0.1234567890123456,0.1234567890123457,3.14159265358979,92.265932524778194,92.265932524778179,
   -1.6611623539523179,-1.6611623539523159,999.5,0.30000000000000004,0.1,1e-5,2.5e-7,64.5487,64.56,19.0000005,19.0};
 String p=a[0]; new java.io.File(p).delete(); DoubleDeXOREncoder e=new DoubleDeXOREncoder(p); for(double x:v) e.encode(x); e.flush();
 StringBuilder s=new StringBuilder(); for(byte b:Files.readAllBytes(Paths.get(p))) s.append(String.format("%02x",b)); System.out.println(s);
 DoubleDeXORDecoder d=new DoubleDeXORDecoder(p); StringBuilder t=new StringBuilder();
 for(int i=0;i<v.length;i++){ t.append(String.format("UINT64_C(0x%016x),", Double.doubleToRawLongBits(d.decodeDouble()))); } System.out.println(t);
 int inexact=0; DoubleDeXORDecoder d2=new DoubleDeXORDecoder(p); for(double x:v) if(Double.doubleToRawLongBits(d2.decodeDouble())!=Double.doubleToRawLongBits(x)) inexact++; System.out.println("inexact="+inexact);
}}
