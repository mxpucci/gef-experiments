import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import org.urbcomp.startdb.selfstar.compressor.*;
import org.urbcomp.startdb.selfstar.compressor.xor.*;
import org.urbcomp.startdb.selfstar.decompressor.*;
import org.urbcomp.startdb.selfstar.decompressor.xor.*;
public class Audit {
 public static void main(String[] args)throws Exception {
  for(String f:args){
   ByteBuffer in=ByteBuffer.wrap(Files.readAllBytes(Paths.get(f))).order(ByteOrder.LITTLE_ENDIAN);
   double[] v=new double[in.remaining()/8];for(int i=0;i<v.length;i++)v[i]=in.getDouble();
   SElfStarCompressor s=new SElfStarCompressor(new SElfStarXORCompressor());
   SElfStarDecompressor sd=new SElfStarDecompressor(new SElfStarXORDecompressor());
   long et=0,st=0,em=0,sm=0;
   try(DataOutputStream ef=new DataOutputStream(new FileOutputStream(f+".java.elf"));DataOutputStream sf=new DataOutputStream(new FileOutputStream(f+".java.self"))){
    for(int p=0;p<v.length;p+=1000){
     ElfStarCompressor e=new ElfStarCompressor(new ElfStarXORCompressor());
     ElfStarDecompressor ed=new ElfStarDecompressor(new ElfStarXORDecompressor());
     int n=Math.min(1000,v.length-p); for(int i=0;i<n;i++){e.addValue(v[p+i]);s.addValue(v[p+i]);}
     e.close();s.close();byte[] eb=e.getBytes(),sb=s.getBytes();et+=eb.length;st+=sb.length;
     ef.writeInt(eb.length);ef.write(eb);sf.writeInt(sb.length);sf.write(sb);
     ed.setBytes(eb);sd.setBytes(sb);List<Double> ev=ed.decompress(),sv=sd.decompress();
     if(ev.size()!=n||sv.size()!=n)throw new RuntimeException("length");
     for(int i=0;i<n;i++){if(Double.doubleToRawLongBits(ev.get(i))!=Double.doubleToRawLongBits(v[p+i]))em++;if(Double.doubleToRawLongBits(sv.get(i))!=Double.doubleToRawLongBits(v[p+i]))sm++;}
     s.refresh();sd.refresh();
    }
   }
   System.out.println(f+","+v.length+","+et+","+st+","+em+","+sm);
  }
 }
}
