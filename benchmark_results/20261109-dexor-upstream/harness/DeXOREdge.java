import java.io.*; import java.nio.file.*; import java.util.*;
import algorithms.DeXOR.encoder.DoubleDeXOREncoder;
import algorithms.DeXOR.decoder.DoubleDeXORDecoder;
// Edge/fuzz fixture: per vector writes int32 n, n doubles (raw bits, BE), int32 len (-1 if Java threw), bytes,
// then n decoded doubles from the stock Java decoder (only when len>=0).
public class DeXOREdge {
 static List<double[]> vectors(){
  List<double[]> r=new ArrayList<>();
  r.add(new double[]{1,2,3,10,11,11,11,-11,-12,0,0.5,0.25,1.2345678901234567,1.2345678901234567,0.25});
  r.add(new double[]{0.1234567890123456,0.1234567890123457,1.2345678901234e-14,3.14159265358979,2.718281828459045,1.23456789e-8,1.5e-14,0.000123456789012345,42.0,42.5,1e-3,9.999999e6,1e7,1.25e7,123456.789});
  r.add(new double[]{0.0,-0.0,0.0,1.0,-1.0,-0.0,0.1,-0.1,0.2,-0.30000000000000004});
  r.add(new double[]{1.5,Double.NaN,2.5,3.5});
  r.add(new double[]{19.0000005,19.0,19.0000005,64.5487,64.56,88.1537,88.1479});
  r.add(new double[]{1e8,1e9+1,123456789012.0,5e11,999999999999.0,1.0,2.0});
  r.add(new double[]{1e-30,2e-30,1.0});
  r.add(new double[]{Double.POSITIVE_INFINITY,1.0});
  r.add(new double[]{Double.MIN_VALUE,1.0});
  Random g=new Random(12345);
  for(int k=0;k<3000;k++){
   int n=1+g.nextInt(60); double[] v=new double[n]; int kind=k%8; int dec=g.nextInt(17); double base=(g.nextDouble()-0.5)*Math.pow(10,g.nextInt(12)-3);
   for(int i=0;i<n;i++){
    switch(kind){
     case 0: v[i]=Math.round((base+g.nextGaussian())*Math.pow(10,dec))/Math.pow(10,dec); break;
     case 1: v[i]=(double)(g.nextInt(2000001)-1000000)/Math.pow(10,dec%10); break;
     case 2: v[i]=g.nextDouble()*Math.pow(10,g.nextInt(30)-15); break;
     case 3: v[i]=Double.longBitsToDouble(g.nextLong()); break;
     case 4: v[i]=Math.round(g.nextGaussian()*1e6)/1e2 + (i%7==0?0.0:0.001); break;
     case 5: v[i]=(g.nextInt(3)-1)*Math.round(g.nextDouble()*1e4)/Math.pow(10,g.nextInt(16)); break;
     case 6: v[i]=base + i*Math.pow(10,-dec); break;
     default: v[i]=(double)g.nextLong()/Math.pow(10,g.nextInt(20)); break;
    }
   }
   r.add(v);
  }
  return r;
 }
 public static void main(String[] a) throws Exception {
  String tmp=a[1]; int thrown=0,total=0;
  try(DataOutputStream o=new DataOutputStream(new BufferedOutputStream(new FileOutputStream(a[0])))){
   for(double[] v:vectors()){total++;
    o.writeInt(v.length); for(double x:v) o.writeLong(Double.doubleToRawLongBits(x));
    String p=tmp+"/edge.bin"; new File(p).delete(); byte[] bs=null; double[] dec=new double[v.length];
    boolean decThrew=false;
    try{ DoubleDeXOREncoder e=new DoubleDeXOREncoder(p); for(double x:v) e.encode(x); e.flush(); bs=Files.readAllBytes(Paths.get(p));
         decThrew=true; DoubleDeXORDecoder d=new DoubleDeXORDecoder(p); for(int i=0;i<v.length;i++) dec[i]=d.decodeDouble(); decThrew=false; }
    catch(RuntimeException ex){ if(!decThrew) bs=null; thrown++; }
    if(bs==null){o.writeInt(-1);} else if(decThrew){o.writeInt(-2);o.writeInt(bs.length);o.write(bs);} else {o.writeInt(bs.length);o.write(bs);for(double x:dec)o.writeLong(Double.doubleToRawLongBits(x));}
   }
  }
  System.out.println("vectors="+total+" java_threw="+thrown);
 }
}
