import java.io.*;
import yyy.ts.compress.camel.Camel;
import yyy.ts.compress.camel.CamelDecompressor;
public class CamelDiagnostics {
 static long read(DataInputStream in)throws IOException{return Long.reverseBytes(in.readLong());}
 public static void main(String[] args)throws Exception{
  for(String path:args)try(DataInputStream in=new DataInputStream(new BufferedInputStream(new FileInputStream(path)))){
   long n=read(in),dec=read(in),scale=1;for(int d=0;d<dec;d++)scale*=10;
   int found=0;
   for(long start=0;start<n && found<5;start+=1000){
    int count=(int)Math.min(1000,n-start);double[] data=new double[count];Camel c=new Camel();
    for(int i=0;i<count;i++){data[i]=(double)read(in)/(double)scale;c.addValue(data[i]);}c.close();
    CamelDecompressor decoder=new CamelDecompressor(c.getOut());
    for(int i=0;i<count;i++){double value=decoder.readPair();if(Double.doubleToRawLongBits(value)!=Double.doubleToRawLongBits(data[i])){
     System.out.println(path+","+(start+i)+","+dec+","+Double.toHexString(data[i])+","+Double.toHexString(value)+","+data[i]+","+value);found++;if(found>=5)break;
    }}
   }
  }
 }
}
