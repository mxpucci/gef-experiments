package algorithms.DeXOR.encoder;
import algorithms.DeXOR.DeXORTools;
// DIAGNOSTIC ONLY (not used by the benchmark): upstream Native encoder with one change --
// a decimal encoding is accepted only if the upstream decoder's arithmetic reproduces the
// value bit-for-bit (in addition to upstream's comp(...) tolerance test). Everything else
// is copied verbatim from DoubleDeXOREncoder.Method.Decimal_XOR at 06128b3.
public class ExactDeXOREncoder extends DoubleDeXOREncoder {
    private double cachedAlpha = 0; // decoder's previous_alpha
    public ExactDeXOREncoder(String outputPath) { super(outputPath); this.method = new ExactNative(); }
    protected class ExactNative extends Method {
        @Override
        protected void Decimal_XOR(double value) {
            int q = DeXORTools.getEnd(value, previous_q);
            int delta = 0;
            double alpha = 0;
            while (delta < 16) {
                double pow = DeXORTools.getP10(q + delta);
                long a = DeXORTools.truncate(value / pow);
                long b = DeXORTools.truncate(previous_value / pow);
                if (a == b) { alpha = a * pow; break; }
                delta++;
            }
            double pow = DeXORTools.getP10(q);
            double residual = value - alpha;
            long beta = Math.round((residual) / pow);
            boolean exact = false;
            if (delta < 16) {
                boolean reuse = q == previous_q && delta == previous_delta;
                double dAlpha = reuse ? cachedAlpha : alpha;
                long sign = dAlpha > 0 ? 1 : -1;
                if (DeXORTools.comp(dAlpha, 0) == 0) sign = value > 0 ? 1 : -1;
                double rec = dAlpha + (sign * Math.abs(beta)) * DeXORTools.getP10(q);
                exact = Double.doubleToRawLongBits(rec) == Double.doubleToRawLongBits(value);
            }
            if (delta >= 16 || DeXORTools.comp(alpha + beta * pow, value, pow) != 0 || !exact) {
                out.write(true); out.write(true); ExceptionHandle(value); return;
            }
            beta = Math.abs(beta);
            boolean flag = q == previous_q;
            if (flag && delta == previous_delta) { out.write(true); out.write(false); }
            else {
                out.write(false); out.write(flag);
                if (!flag) { out.write(q + 20, 5); previous_q = q; }
                out.write(delta, 4); previous_delta = delta; cachedAlpha = alpha;
            }
            if (DeXORTools.comp(alpha, 0) == 0) out.write(value > 0);
            out.write(beta, DeXORTools.decimalBits(delta));
            previous_value = value;
        }
    }
}
