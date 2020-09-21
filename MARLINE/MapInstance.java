package msbc;

import com.yahoo.labs.samoa.instances.Instance;
import weka.core.matrix.Matrix;

//import findPoint.Matrix;

public class MapInstance {
//    protected Matrix tmatrix;
//
//    public MapInstance(){
//        tmatrix = null;
//        try {
//        tmatrix = new Matrix();
//    } catch (MWException e) {
//        e.printStackTrace();
//    }
//    }

    public Instance getMappedInstance(double[][] centroid, double[][] target_centroid, Instance inst){
        Instance tmp_inst = inst.copy();
        double[] sc_vector = getVector(centroid);
        double[] tc_vector = getVector(target_centroid);

        double[][] target_points = new double[2][inst.numInputAttributes()];
        target_points[0] = target_centroid[0].clone();
        for (int i = 0; i < inst.numInputAttributes(); i ++){
            target_points[1][i] = inst.valueInputAttribute(i);
        }

        double[] t_vector = getVector(target_points);

        Matrix sm = new Matrix(sc_vector, sc_vector.length);
        Matrix tm = new Matrix(tc_vector, tc_vector.length);
        Matrix tv = new Matrix(t_vector, 1);
        Matrix u = sm.times(1/sm.norm2());
        Matrix v = tm.times(1/tm.norm2());
//        System.out.println("sm: " + sm);
//        System.out.println("tm: " + tm);
//        System.out.println("u: " + u);
//        System.out.println("v: " + v);

        Matrix s = reflection(Matrix.identity(sm.getRowDimension(),sm.getRowDimension()), u.plus(v));
        Matrix r = reflection(s, v);

        Matrix t = tv.times(r).times(sm.norm2()/tm.norm2());
        double[] a = centroid[0];
        Matrix result = t.plus(new Matrix(a,1));

        double[] point = result.getColumnPackedCopy();
        for (int i = 0; i < tmp_inst.numInputAttributes(); i ++){
            tmp_inst.setValue(i, point[i]);
        }
        return tmp_inst;
    }

    public Matrix reflection(Matrix u, Matrix n){
//        Matrix a = n.transpose().times(u);
//        Matrix b = n.times(a);
//        Matrix c = b.times(2);
//        //System.out.println("n: " + n);
//        Matrix d = n.transpose().times(n);
//        //System.out.println("d: " + d);
//        Matrix d1 = d.inverse();
//        //System.out.println("d1: " + d1);
//        Matrix e = c.times(d.inverse().get(0,0));
//        //Matrix e = d.inverse().times(c);
//        Matrix f = u.minus(e);

        Matrix v = u.minus(n.times(2).times(n.transpose().times(u)).times(n.transpose().times(n).inverse().get(0,0)));
        return v;
    }
//    public Instance getMappedInstance(double[][] centroid, double[][] target_centroid, Instance inst){
//        Instance tmp_inst = inst.copy();
//        double[] sc_vector = getVector(centroid);
//        double[] tc_vector = getVector(target_centroid);
////        System.out.println("s: (" + centroid[0][0] + ","+ centroid[0][1] + ") (" + centroid[1][0] + "," + centroid[1][1] + ")");
////        System.out.println("t: (" + target_centroid[0][0] + "," + target_centroid[0][1] + ") (" + target_centroid[1][0]  + "," + target_centroid[1][1] + ")");
//
//        double[][] target_points = new double[2][inst.numInputAttributes()];
//        target_points[0] = target_centroid[0].clone();
//        for (int i = 0; i < inst.numInputAttributes(); i ++){
//            target_points[1][i] = inst.valueInputAttribute(i);
//        }
//
//        double[] t_vector = getVector(target_points);
//
//        Object[] result = new Object[0];
//        try {
//            result = tmatrix.findPoint(1,sc_vector, tc_vector, t_vector, centroid[0]);
//        } catch (MWException e) {
//            e.printStackTrace();
//        }
//
//        MWNumericArray temp = (MWNumericArray)result[0];
//        double[][] point = (double[][])temp.toDoubleArray();
//
//        //System.out.println(inst.valueInputAttribute(7));
//
//
//        for (int i = 0; i < tmp_inst.numInputAttributes(); i ++){
//            tmp_inst.setValue(i, point[0][i]);
//        }
//
//        return tmp_inst;
//    }

    protected static double[] getVector(double[][] points){
        double[] result = new double[points[0].length];
        for (int i = 0; i < points[0].length; i++){
            result[i] = points[1][i] - points[0][i];
        }

        return result;
    }
}
