package msbc;

import Others.MultiSourceInstance;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.WEKAClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MSBC extends AbstractClassifier implements MultiClassClassifier {
    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class,"meta.OzaBoost");
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
            "Drift detection method to use.", ChangeDetector.class, "HDDM_A_Test");
    public IntOption targetDomainIndexOption = new IntOption("targetDomainIndex", 't',
            "The Index of target domain in data set", 1, 0, Integer.MAX_VALUE);
    public FloatOption forgettingFactor = new FloatOption("forgettingFactor", 'f',
            "Forgetting Factor.", 0.9, 0.0, 1.0);
    public FloatOption accuracyOption = new FloatOption("accuracy", 'a',
            "Minimum fraction of weight per model.", 0.0, 0.0, 1.0);

    protected Map<Integer, List<BaseLearner>> ensemble;
    public static final int DDM_OUTCONTROL_LEVEL = 2;
    protected MapInstance mp;

    public int changeDetected = 0;

    @Override
    public void resetLearningImpl() {
        ensemble = new HashMap<>();
        mp = new MapInstance();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        MultiSourceInstance instance = (MultiSourceInstance)inst;
        int id = instance.getDataId();
        if (!ensemble.containsKey(id)){
            BaseLearner bl = new BaseLearner();
            bl.baseLearnerOption = (ClassOption) this.baseLearnerOption.copy();
            bl.driftDetectionMethodOption = (ClassOption) this.driftDetectionMethodOption.copy();
            bl.forgettingFactor = (FloatOption) this.forgettingFactor.copy();
            bl.prepareForUse();
            bl.resetLearning();
            List<BaseLearner> bl_list = new ArrayList<>();
            bl_list.add(bl);
            ensemble.put(id, bl_list);
        }

        //detect drift
        int size = ensemble.get(id).size();
        int ddmLevel = ensemble.get(id).get(size - 1).detectDrift(inst);
        if (ddmLevel == DDM_OUTCONTROL_LEVEL) {
            this.changeDetected++;
            BaseLearner newclassifier = (BaseLearner) ensemble.get(id).get(size - 1).getNewclassifier().copy();
            if (newclassifier.getClassifier() instanceof WEKAClassifier) {
                ((WEKAClassifier) newclassifier.getClassifier()).buildClassifier();
            }
            ensemble.get(id).add(newclassifier);
            size = ensemble.get(id).size();
        }

        ensemble.get(id).get(size-1).trainOnInstanceImpl(inst);

        //reset each sub-classifier's performance
        if (ddmLevel == DDM_OUTCONTROL_LEVEL&&id == this.targetDomainIndexOption.getValue()) {
            for (Map.Entry<Integer, List<BaseLearner>> entry : ensemble.entrySet()) {
                for (BaseLearner bl : entry.getValue()) {
                    bl.resetPerformance();
                    bl.resetWeights();
                }
            }
        }

        if (id == this.targetDomainIndexOption.getValue()) {
            if (ensemble.get(id).get(size - 1).getCentroid() != null) {
                double lossc = 0.0;
                double lossw = 0.0;
                Map<Integer, List<DoubleVector[]>> votes_list = new HashMap<>();
                for (Map.Entry<Integer, List<BaseLearner>> entry : ensemble.entrySet()) {
                    votes_list.put(entry.getKey(), new ArrayList<>());
                    for (BaseLearner bl : entry.getValue()) {
                        if (bl.getCentroid() != null) {
                            //Instance instExp = inst.copy();
                            Instance instExp = bl.equals(ensemble.get(id).get(size - 1))
                                    ? inst.copy() : mp.getMappedInstance(bl.getCentroid(), ensemble.get(id).get(size - 1).getCentroid(), inst);
                            Classifier[] sub_classifiers = bl.getClassifier().getSubClassifiers();
                            DoubleVector[] votes = new DoubleVector[sub_classifiers.length];
                            double[] weights = bl.getPerformance();
                            for (int i = 0; i < sub_classifiers.length; i++) {
                                votes[i] = new DoubleVector(sub_classifiers[i].getVotesForInstance(instExp));
                                if (votes[i].sumOfValues() > 0.0) {
                                    votes[i].normalize();
                                    votes[i].scaleValues(weights[i]);
                                    lossw += votes[i].getValue((1 - (int)inst.classValue()));
                                    lossc += votes[i].getValue((int) inst.classValue());
                                }
                            }
                            votes_list.get(entry.getKey()).add(votes);
                        }else
                            votes_list.get(entry.getKey()).add(null);
                    }
                }

                for (Map.Entry<Integer, List<BaseLearner>> entry : ensemble.entrySet()) {
                    for (BaseLearner bl : entry.getValue()) {
                        if (bl.getCentroid() != null) {
//                            System.out.println("size1: " + entry.getValue().size());
//                            System.out.println("size: " + votes_list.get(entry.getKey()).size());
//                            System.out.println(entry.getKey()+ " : "+ entry.getValue().indexOf(bl));
                            DoubleVector[] votes = votes_list.get(entry.getKey()).get(entry.getValue().indexOf(bl));
                            bl.updatePerformance(votes, inst, lossc, lossw);
                        }
                    }
                }

                double sum_alpha = 0.0;
                for (Map.Entry<Integer, List<BaseLearner>> entry: ensemble.entrySet()) {
                    for (BaseLearner bl : entry.getValue()) {
                        double[] alpha = bl.getPerformance();
                        for (int i = 0; i < alpha.length; i++){
                            if (alpha[i] > accuracyOption.getValue())
                                sum_alpha +=  alpha[i];
                        }
                    }
                }

                for (Map.Entry<Integer, List<BaseLearner>> entry: ensemble.entrySet()) {
                    for (BaseLearner bl : entry.getValue()) {
                        double[] alpha = bl.getPerformance();
                        double[] weights = new double[alpha.length];
                        for (int i = 0; i < alpha.length; i++){
                            if (alpha[i] > accuracyOption.getValue())
                                weights[i] =  alpha[i]/sum_alpha;
                            else
                                weights[i] = 0;
                        }
                        bl.setWeights(weights);
                    }
                }
            }
        }
    }


//    int count = 0;
//    public double source_weight = 0.0;
//    public double pt_weight = 0.0;
//    public double ct_weight = 0.0;
//    public DoubleVector sourceVote;
//    public DoubleVector ptVote;
//    public DoubleVector ctVote;



    double weightR;
    @Override
    public double[] getVotesForInstance(Instance inst) {



//        count++;
//        if (count == 5000){
//            source_weight = 0.0;
//            pt_weight = 0.0;
//            ct_weight = 0.0;
//        }
//
//        sourceVote = new DoubleVector();
//        ptVote = new DoubleVector();
//        ctVote = new DoubleVector();

        double sweight = 0.0;
        double tweight = 0.0;
        weightR = 0.0;


        int id = targetDomainIndexOption.getValue();
        if (!ensemble.containsKey(id)){
            double[] a = {0.0,0.0};
            return a;
        }
        int size = ensemble.get(id).size();

        BaseLearner tc = ensemble.get(id).get(size - 1);
        if (tc.getExampleCount()[0]<1 && tc.getExampleCount()[1]>=1){
            double[] r = new double[2];
            r[0] = 0.0;
            r[1] = 1.0;
            return r;
        }else if(tc.getExampleCount()[0] >=1 && tc.getExampleCount()[1] <1){
            double[] r = new double[2];
            r[0] = 1.0;
            r[1] = 0.0;
            return r;
        }else if(tc.getExampleCount()[0] <1 && tc.getExampleCount()[1] <1) {
            double[] r = new double[2];
            r[0] = 0.0;
            r[1] = 0.0;
            return r;
        }

        DoubleVector combinedVote = new DoubleVector();
        for (Map.Entry<Integer, List<BaseLearner>> entry: ensemble.entrySet()) {
            for (BaseLearner bl : entry.getValue()) {
                if (bl.getCentroid() != null) {
                    //Instance instExp = inst.copy();
                    Instance instExp = bl.equals(tc)
                            ? inst.copy() : mp.getMappedInstance(bl.getCentroid(), tc.getCentroid(), inst);
                    Classifier[] sub_classifiers = bl.getClassifier().getSubClassifiers();
                    double[] weights = bl.getWeights();
                    for (int i = 0; i < sub_classifiers.length; i++){
                        if (weights[i] > 0){
                            DoubleVector vote = new DoubleVector(sub_classifiers[i].getVotesForInstance(instExp));
                            //System.out.println(entry.getKey()+ " : " + entry.getValue().indexOf(bl) + " : " +  i + " : " + vote + " label: " + inst.classValue() + " w: " + weights[i] + " Num: " + (bl.getExampleCount()[0] + bl.getExampleCount()[1])+ " c: " + bl.sc[i] + " : "+ bl.sw[i]);
                            if (vote.sumOfValues() > 0.0) {
                                vote.normalize();

                                if (!bl.equals(tc)){
                                    sweight += weights[i];
                                }else
                                    tweight += weights[i];
//                                if (entry.getKey() == 0){
//                                    source_weight += weights[i];
//                                    sourceVote.addValues(vote);
//                                }
//                                else if (!bl.equals(tc)) {
//                                    pt_weight += weights[i];
//                                    //System.out.println(vote);
//                                    ptVote.addValues(vote);
//                                }
//                                else {
//                                    ct_weight += weights[i];
//                                    ctVote.addValues(vote);
//                                }
                                vote.scaleValues(weights[i]);
                                //System.out.println(entry.getKey()+ " : " + entry.getValue().indexOf(bl) + " : " +  i + " : " + vote + " label: " + inst.classValue() + " w: " + weights[i] + " Num: " + (bl.getExampleCount()[0] + bl.getExampleCount()[1])+ " c: " + bl.sc[i] + " : "+ bl.sw[i]);
                                //System.out.println(entry.getKey()+ " : " + entry.getValue().indexOf(bl) + " : " +  i + " : " + weights[i]);
                                combinedVote.addValues(vote);

                            }
                        }
                    }
                }
            }
        }
//        System.out.println(sweight/(sweight+tweight));
        weightR = sweight/(sweight+tweight);
        combinedVote.normalize();
        return combinedVote.getArrayRef();
    }


    public double getWeightR() {
        return weightR;
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
}
