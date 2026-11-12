import java.io.*;
import java.util.*;
import yyy.ts.compress.camel.Camel;
import yyy.ts.compress.camel.CamelDecompressor;

// Harness only: the codec is the pinned authors' Java implementation.
public class CamelBenchmark {
 static volatile double sink;
 static long readLong(DataInputStream in) throws IOException { return Long.reverseBytes(in.readLong()); }
 static byte[] encode(double[] data,int start,int end) {
  Camel c=new Camel();for(int i=start;i<end;i++)c.addValue(data[i]);c.close();
  // Use the real bit writer: Camel.getSize() omits integer sign bits.
  return Arrays.copyOf(c.getOut(),Math.toIntExact((c.getOutputStream().writtenBits()+7)/8));
 }
 static double point(byte[] bytes,int offset) {
  CamelDecompressor d=new CamelDecompressor(bytes);double value=0;
  for(int i=0;i<=offset;i++)value=d.readPair();return value;
 }
 public static void main(String[] args) throws Exception {
  double[] data;long[] points;int[] ranges;long[][] starts;
  try(DataInputStream in=new DataInputStream(new BufferedInputStream(new FileInputStream(args[0])))) {
   int n=Math.toIntExact(readLong(in));data=new double[n];
   for(int i=0;i<n;i++)data[i]=Double.longBitsToDouble(readLong(in));
   int q=Math.toIntExact(readLong(in));points=new long[q];for(int i=0;i<q;i++)points[i]=readLong(in);
   int r=Math.toIntExact(readLong(in));ranges=new int[r];starts=new long[r][];
   for(int k=0;k<r;k++){ranges[k]=Math.toIntExact(readLong(in));int count=Math.toIntExact(readLong(in));starts[k]=new long[count];for(int i=0;i<count;i++)starts[k][i]=readLong(in);}
  }
  final int block=1000,n=data.length;
  // Warm up original encoder/decoder outside measurements, retaining final tails.
  for(int i=0;i<100 && n>0;i++){int start=(i%((n+block-1)/block))*block;int end=Math.min(start+block,n);byte[] b=encode(data,start,end);sink=point(b,end-start-1);}
  byte[][] encoded=new byte[(n+block-1)/block][];long bytes=0;
  long t=System.nanoTime();
  for(int i=0;i<encoded.length;i++){encoded[i]=encode(data,i*block,Math.min((i+1)*block,n));bytes+=encoded[i].length;}
  long comp=System.nanoTime()-t;
  double[] decoded=new double[n];t=System.nanoTime();
  for(int i=0;i<encoded.length;i++){CamelDecompressor d=new CamelDecompressor(encoded[i]);for(int j=i*block;j<Math.min((i+1)*block,n);j++)decoded[j]=d.readPair();}
  long decomp=System.nanoTime()-t,mismatches=0;
  for(int i=0;i<n;i++)if(Double.doubleToRawLongBits(data[i])!=Double.doubleToRawLongBits(decoded[i]))mismatches++;
  double sum=0;t=System.nanoTime();
  for(long index:points)sum+=point(encoded[(int)(index/block)],(int)(index%block));
  long ra=System.nanoTime()-t;sink=sum;
  for(int q=0;q<Math.min(32,points.length);q++) {
   int index=(int)points[q];
   if(Double.doubleToRawLongBits(point(encoded[index/block],index%block))!=Double.doubleToRawLongBits(decoded[index]))throw new IllegalStateException("point/full decode disagreement");
  }
  try(PrintWriter out=new PrintWriter(args[1])) {
   out.println((bytes*8)+" "+comp+" "+decomp+" "+mismatches+" "+ra+" "+points.length+" "+ranges.length);
   for(int k=0;k<ranges.length;k++) {
    double[] buffer=new double[ranges[k]];sum=0;t=System.nanoTime();
    for(long startLong:starts[k]) {
     int start=(int)startLong,end=start+ranges[k],pos=0;
     for(int b=start/block;b<=(end-1)/block;b++) {
      CamelDecompressor d=new CamelDecompressor(encoded[b]);int lo=Math.max(start,b*block),hi=Math.min(end,(b+1)*block);
      for(int j=b*block;j<lo;j++)d.readPair();
      for(int j=lo;j<hi;j++)buffer[pos++]=d.readPair();
     }
     sum+=buffer[0];
    }
    long elapsed=System.nanoTime()-t;sink=sum;
    out.println(ranges[k]+" "+elapsed+" "+starts[k].length);
    if(starts[k].length>0) {
     int start=(int)starts[k][starts[k].length-1];
     for(int j=0;j<buffer.length;j++)if(Double.doubleToRawLongBits(buffer[j])!=Double.doubleToRawLongBits(decoded[start+j]))throw new IllegalStateException("range/full decode disagreement");
    }
   }
  }
 }
}
