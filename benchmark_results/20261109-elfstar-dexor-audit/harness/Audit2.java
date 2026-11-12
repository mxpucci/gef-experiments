import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.urbcomp.startdb.selfstar.compressor.*;
import org.urbcomp.startdb.selfstar.compressor.xor.*;
import org.urbcomp.startdb.selfstar.decompressor.*;
import org.urbcomp.startdb.selfstar.decompressor.xor.*;
import algorithms.DeXOR.encoder.DoubleDeXOREncoder;
import algorithms.DeXOR.decoder.DoubleDeXORDecoder;
// args: in.f64 outprefix tmpdir. Writes outprefix.{elfstar,nohuff,dexor}: per block int32 len + bytes.
// Prints: elfstar_bytes, nohuff_bytes, dexor_bytes, and bitwise round-trip mismatches for each.
public class Audit2 {
 static double[][] load(String f) throws IOException {
  ByteBuffer in=ByteBuffer.wrap(Files.readAllBytes(Paths.get(f))).order(ByteOrder.LITTLE_ENDIAN);
  int nb=in.getInt(); double[][] r=new double[nb][];
  for(int b=0;b<nb;b++){int n=in.getInt();r[b]=new double[n];for(int i=0;i<n;i++)r[b][i]=in.getDouble();}
  return r;
 }
 static boolean same(double a,double b){return Double.doubleToRawLongBits(a)==Double.doubleToRawLongBits(b);}
 public static void main(String[] a) throws Exception {
  double[][] blocks=load(a[0]); String tmp=a[2];
  long eb=0,nb=0,db=0,em=0,nm=0,dm=0,dmApprox=0;
  try(DataOutputStream eo=new DataOutputStream(new FileOutputStream(a[1]+".elfstar"));
      DataOutputStream no=new DataOutputStream(new FileOutputStream(a[1]+".nohuff"));
      DataOutputStream xo=new DataOutputStream(new FileOutputStream(a[1]+".dexor"));
      DataOutputStream xl=new DataOutputStream(new FileOutputStream(a[1]+".dexor_bits"));
      DataOutputStream xd=new DataOutputStream(new FileOutputStream(a[1]+".dexor_dec"))){
   for(double[] v:blocks){
    ElfStarCompressor e=new ElfStarCompressor(new ElfStarXORCompressor());
    ElfStarCompressorNoHuff h=new ElfStarCompressorNoHuff(new ElfStarXORCompressor());
    for(double x:v){e.addValue(x);h.addValue(x);}
    e.close();h.close();byte[] ebs=e.getBytes(),hbs=h.getBytes();
    eo.writeInt(ebs.length);eo.write(ebs);no.writeInt(hbs.length);no.write(hbs);eb+=ebs.length;nb+=hbs.length;
    ElfStarDecompressor ed=new ElfStarDecompressor(new ElfStarXORDecompressor());ed.setBytes(ebs);
    List<Double> ev=ed.decompress(); if(ev.size()!=v.length) throw new RuntimeException("elf* len");
    for(int i=0;i<v.length;i++) if(!same(ev.get(i),v[i])) em++;
    ElfStarDecompressorNoHuff hd=new ElfStarDecompressorNoHuff(new ElfStarXORDecompressor());hd.setBytes(hbs);
    List<Double> hv=hd.decompress(); if(hv.size()!=v.length) throw new RuntimeException("nohuff len "+hv.size());
    for(int i=0;i<v.length;i++) if(!same(hv.get(i),v[i])) nm++;
    String p=tmp+"/dexor.bin"; new File(p).delete();
    DoubleDeXOREncoder de=new DoubleDeXOREncoder(p); for(double x:v) xl.writeInt(de.encode(x)); de.flush();
    byte[] xbs=Files.readAllBytes(Paths.get(p)); xo.writeInt(xbs.length);xo.write(xbs);db+=xbs.length;
    DoubleDeXORDecoder dd=new DoubleDeXORDecoder(p);
    for(int i=0;i<v.length;i++){double y=dd.decodeDouble(); xd.writeLong(Double.doubleToRawLongBits(y)); if(!same(y,v[i])){dm++; if(Math.abs(y-v[i])<=1e-9*Math.max(1,Math.abs(v[i])))dmApprox++;}}
   }
  }
  System.out.println(a[0]+",java,"+blocks.length+","+eb+","+nb+","+db+","+em+","+nm+","+dm+","+dmApprox);
 }
}
