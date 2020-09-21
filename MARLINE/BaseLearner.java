package msbc;

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.meta.WEKAClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

public class BaseLearner extends AbstractClassifier implements MultiClassClassifier {

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class,"meta.OzaBoost");
    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
            "Drift detection method to use.", ChangeDetector.class, "HDDM_A_Test");
    public FloatOption forgettingFactor = new FloatOption("forgettingFactor", 'f',
            "Forgetting Factor.", 0.9, 0.0, 1.0);

    protected Classifier classifier;
    protected BaseLearner newclassifier;
    protected ChangeDetector changeDetector;
    protected double[][] centroid;
    protected double[] example_count;
    protected double[] performance;
    protected double[] weights;
    protected double[] sc;
    protected double[] sw;
    protected int ddmLevel;
    protected boolean newClassifierReset;
    public static final int DDM_INCONTROL_LEVEL = 0;
    public static final int DDM_WARNING_LEVEL = 1;
    public static final int DDM_OUTCONTROL_LEVEL = 2;

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return classifier.getVotesForInstance(inst);
    }

    @Override
    public void resetLearningImpl() {
        classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        classifier.resetLearning();
        changeDetector = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
        newClassifierReset = false;
        centroid = new double[2][];
        example_count = new double[2];
        example_count[0] = 0;
        example_count[1] = 0;
        performance = new double[classifier.getSubClassifiers().length];
        sc = new double[classifier.getSubClassifiers().length];
        sw = new double[classifier.getSubClassifiers().length];
        this.resetPerformance();
        weights = new double[classifier.getSubClassifiers().length];
        this.resetWeights();
        //newclassifier = (BaseLearner) this.copy();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        example_count[(int) inst.classValue()] = example_count[(int) inst.classValue()]*forgettingFactor.getValue() + 1;
        updateCentroid(inst);
        classifier.trainOnInstance(inst);
    }

    public int detectDrift(Instance inst){
        boolean prediction = classifier.correctlyClassifies(inst);
        changeDetector.input(prediction ? 0.0 : 1.0);
        ddmLevel = DDM_INCONTROL_LEVEL;
        if (changeDetector.getChange()) {
            this.ddmLevel =  DDM_OUTCONTROL_LEVEL;
        }
        if (changeDetector.getWarningZone()) {
            this.ddmLevel =  DDM_WARNING_LEVEL;
        }
        switch (this.ddmLevel) {
            case DDM_WARNING_LEVEL:
                //System.out.println("warning");
                if (newClassifierReset == true) {
                    newclassifier = new BaseLearner();
                    newclassifier.baseLearnerOption = (ClassOption) this.baseLearnerOption.copy();
                    newclassifier.driftDetectionMethodOption = (ClassOption) this.driftDetectionMethodOption.copy();
                    newclassifier.forgettingFactor = (FloatOption) this.forgettingFactor.copy();
                    newclassifier.prepareForUse();
                    newclassifier.resetLearning();
                    newClassifierReset = false;
                }
                this.newclassifier.trainOnInstance(inst);
                break;
            case DDM_OUTCONTROL_LEVEL:
                if (newClassifierReset == true){
                    newclassifier = new BaseLearner();
                    newclassifier.baseLearnerOption = (ClassOption) this.baseLearnerOption.copy();
                    newclassifier.driftDetectionMethodOption = (ClassOption) this.driftDetectionMethodOption.copy();
                    newclassifier.forgettingFactor = (FloatOption) this.forgettingFactor.copy();
                    newclassifier.prepareForUse();
                    newclassifier.resetLearning();
                }
                //System.out.println("out");
                break;
            case DDM_INCONTROL_LEVEL:
                newClassifierReset = true;
                break;
            default:
        }

        return ddmLevel;
    }

    protected void updateCentroid(Instance inst){
        if (example_count[(int) inst.classValue()] == 1)
            centroid[(int) inst.classValue()] = new double[inst.numInputAttributes()];

        for (int i = 0; i < inst.numInputAttributes(); i ++){
            centroid[(int) inst.classValue()][i] = inst.valueInputAttribute(i) + centroid[(int) inst.classValue()][i]*forgettingFactor.getValue();
        }
    }

    public double[][] getCentroid() {
        if (example_count[0]<1||example_count[1]<1)
            return null;

        double[][] result = new double[centroid.length][];
        for (int i = 0; i < result.length; i++){
            result[i] = new double[centroid[i].length];
            for (int j = 0; j < result[i].length; j++){
                result[i][j] = centroid[i][j]/example_count[i];
            }
        }
        return result;
    }

    public double[] getExampleCount() {
        return example_count;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    public BaseLearner getNewclassifier() {
        return newclassifier;
    }

    public int getDdmLevel() {
        return ddmLevel;
    }

    public void updatePerformance(DoubleVector[] votes, Instance inst, double lossc, double lossw){
        for (int i = 0; i < votes.length; i++){
            if (votes[i].sumOfValues() > 0){
                sw[i] = sw[i]*forgettingFactor.getValue() + ((lossw+Double.MIN_VALUE)/(lossc + Double.MIN_VALUE)) * (votes[i].getValue(1 - (int)inst.classValue())/(lossw+Double.MIN_VALUE));
                sc[i] = sc[i]*forgettingFactor.getValue() + ((lossw+Double.MIN_VALUE)/(lossc + Double.MIN_VALUE)) * (votes[i].getValue((int) inst.classValue())/(lossc+Double.MIN_VALUE));
            }
        }

        for (int i = 0; i < performance.length; i++){
            performance[i] = (sc[i]+Double.MIN_VALUE)/((sc[i]+Double.MIN_VALUE)+(sw[i]+Double.MIN_VALUE));;
        }
    }

    public double[] getPerformance() {
        return performance;
    }

    public void resetPerformance(){
        for (int i = 0; i < performance.length; i++) {
            performance[i] = 1.0;
            sc[i] = 0.0;
            sw[i] = 0.0;
        }
    }

    public void resetWeights(){
        for (int i = 0; i < weights.length; i++)
            weights[i] = 1.0;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public Classifier getClassifier() {
        return classifier;
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
