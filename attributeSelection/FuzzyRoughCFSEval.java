package weka.attributeSelection;

import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.filters.unsupervised.instance.Resample;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.measure.Measure;
import weka.fuzzy.measure.WeakGamma;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity3;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormLukasiewicz;

import java.util.Enumeration;
import java.util.Vector;
import java.util.BitSet;

public class FuzzyRoughCFSEval
        extends ASEvaluation
        implements SubsetEvaluator, OptionHandler, TechnicalInformationHandler {

    // ... (All fields and globalInfo, getTechnicalInformation, constructor remain the same) ...
    static final long serialVersionUID = -28587895748957681L;

    private Instances m_trainInstances;
    private int m_classIndex;
    private int m_numAttribs;
    private int m_numInstances;

    public Similarity m_Similarity = new Similarity3();
    public Similarity m_DecisionSimilarity;
    public Similarity m_SimilarityEq = new SimilarityEq();
    public TNorm m_TNorm = new TNormLukasiewicz();
    public Implicator m_Implicator = new ImplicatorLukasiewicz();
    public SNorm m_SNorm;

    private double[] m_featureClassDependencies;
    private double[][] m_featureFeatureDependencies;

    private double m_samplePercentage = 100.0;
    private boolean m_unsupervised = false;
    
    public String globalInfo() {
        return "FuzzyRoughCFSEval: Evaluates the worth of a subset of attributes by considering "
                + "the individual predictive ability of each feature and the degree of redundancy between them, "
                + "using Fuzzy Rough Set dependency as the core metric.\n\n"
                + "This method adapts the Correlation-based Feature Selection (CFS) heuristic. "
                + "Instead of using linear correlation, it uses the powerful, non-linear dependency measure (gamma) from "
                + "Fuzzy Rough Set Theory to quantify both feature-class relevance and feature-feature redundancy.\n\n"
                + "For performance on large datasets, dependency calculations can be performed on a stratified sample of the data.\n\n"
                + getTechnicalInformation().toString();
    }
    
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.MISC);
        result.setValue(Field.AUTHOR, "M. A. Hall");
        result.setValue(Field.YEAR, "1999");
        result.setValue(Field.TITLE, "Correlation-based Feature Selection for Machine Learning");
        result.setValue(Field.INSTITUTION, "Department of Computer Science, University of Waikato");
        result.setValue(Field.NOTE, "PhD Thesis");

        TechnicalInformation additional = result.add(Type.ARTICLE);
        additional.setValue(Field.AUTHOR, "R. Jensen, Q. Shen");
        additional.setValue(Field.YEAR, "2009");
        additional.setValue(Field.TITLE, "New Approaches to Fuzzy-rough Feature Selection");
        additional.setValue(Field.JOURNAL, "IEEE Transactions on Fuzzy Systems");

        return result;
    }

    public FuzzyRoughCFSEval() {
        resetOptions();
    }


    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(5);
        newVector.addElement(new Option("\tSimilarity relation.", "R", 1, "-R <spec>"));
        newVector.addElement(new Option("\tT-Norm and S-Norm connectives.", "T", 1, "-T <spec>"));
        newVector.addElement(new Option("\tImplicator.", "I", 1, "-I <spec>"));
        newVector.addElement(new Option("\tPercentage of data to sample for pre-computation (1-100).", "P", 1, "-P <double>"));
        newVector.addElement(new Option("\tTreat selection as unsupervised.", "U", 0, "-U"));

        return newVector.elements();
    }

    /**
     * Parses and sets a given list of options.
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        resetOptions();
        String optionString;

        optionString = Utils.getOption('R', options);
        if (optionString.length() != 0) {
            String[] spec = Utils.splitOptions(optionString);
            if (spec.length == 0) throw new Exception("Invalid Similarity spec string.");
            String className = spec[0];
            spec[0] = ""; // *** THIS IS THE FIX ***
            setSimilarity((Similarity) Utils.forName(Similarity.class, className, spec));
        }

        optionString = Utils.getOption('T', options);
        if (optionString.length() != 0) {
            String[] spec = Utils.splitOptions(optionString);
            if (spec.length == 0) throw new Exception("Invalid TNorm spec string.");
            String className = spec[0];
            spec[0] = ""; // *** THIS IS THE FIX ***
            setTNorm((TNorm) Utils.forName(TNorm.class, className, spec));
        }

        optionString = Utils.getOption('I', options);
        if (optionString.length() != 0) {
            String[] spec = Utils.splitOptions(optionString);
            if (spec.length == 0) throw new Exception("Invalid Implicator spec string.");
            String className = spec[0];
            spec[0] = ""; // *** THIS IS THE FIX ***
            setImplicator((Implicator) Utils.forName(Implicator.class, className, spec));
        }

        optionString = Utils.getOption('P', options);
        if (optionString.length() != 0) {
            setSamplePercentage(Double.parseDouble(optionString));
        }

        setUnsupervised(Utils.getFlag('U', options));
    }


    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        result.add("-R");
        result.add((m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim());
        result.add("-T");
        result.add((m_TNorm.getClass().getName() + " " + Utils.joinOptions(m_TNorm.getOptions())).trim());
        result.add("-I");
        result.add((m_Implicator.getClass().getName() + " " + Utils.joinOptions(m_Implicator.getOptions())).trim());
        result.add("-P");
        result.add(String.valueOf(getSamplePercentage()));
        if (getUnsupervised()) {
            result.add("-U");
        }
        return result.toArray(new String[0]);
    }
    
    // --- Getters and Setters for Options (unchanged) ---
    public void setSimilarity(Similarity s) { m_Similarity = s; }
    public Similarity getSimilarity() { return m_Similarity; }
    public void setTNorm(TNorm t) { m_TNorm = t; m_SNorm = t.getAssociatedSNorm(); }
    public TNorm getTNorm() { return m_TNorm; }
    public void setImplicator(Implicator i) { m_Implicator = i; }
    public Implicator getImplicator() { return m_Implicator; }
    public void setSamplePercentage(double p) { if (p < 1) p = 1; if (p > 100) p = 100; m_samplePercentage = p; }
    public double getSamplePercentage() { return m_samplePercentage; }
    public String samplePercentageTipText() { return "The percentage of data to use for pre-computing dependencies. Lower values speed up the build process significantly on large datasets, at the cost of less accurate estimates."; }
    public void setUnsupervised(boolean u) { m_unsupervised = u; }
    public boolean getUnsupervised() { return m_unsupervised; }
    public String unsupervisedTipText() { return "If true, treats the problem as unsupervised, ignoring the class attribute for relevance calculations."; }

    // --- All other methods (getCapabilities, buildEvaluator, evaluateSubset, etc.) remain unchanged from the previous corrected version ---
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.NUMERIC_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        return result;
    }
    
    @Override
    public void buildEvaluator(Instances data) throws Exception {
        getCapabilities().testWithFail(data);

        m_trainInstances = new Instances(data);
        m_numAttribs = m_trainInstances.numAttributes();
        m_numInstances = m_trainInstances.numInstances();
        m_classIndex = m_trainInstances.classIndex();
        
        if (m_classIndex < 0 || m_unsupervised) {
        	m_unsupervised = true;
        	m_classIndex = -1; // Ensure class is ignored
        	System.err.println("Running in Unsupervised mode. Feature relevance will not be calculated.");
        }

        m_featureClassDependencies = new double[m_numAttribs];
        m_featureFeatureDependencies = new double[m_numAttribs][m_numAttribs];

        Instances dataForComputation = m_trainInstances;
        if (m_samplePercentage < 100.0 && m_numInstances > 0) {
            System.err.println("Sampling data (" + m_samplePercentage + "%)...");

            if (!m_unsupervised) {
                System.err.println("Using stratified sampling.");
                int numFolds = (int) Math.round(100.0 / m_samplePercentage);
                if (numFolds < 2) numFolds = 2;
                
                //System.out.println("Number of folds: "+numFolds);
                StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
                filter.setNumFolds(numFolds);
                filter.setFold(1);
                filter.setInvertSelection(false);
                filter.setInputFormat(m_trainInstances);
                dataForComputation = Filter.useFilter(m_trainInstances, filter);
            } else {
                System.err.println("Using random sampling (unsupervised mode).");
                Resample resample = new Resample();
                resample.setSampleSizePercent(m_samplePercentage);
                resample.setNoReplacement(true);
                resample.setInputFormat(m_trainInstances);
                dataForComputation = Filter.useFilter(m_trainInstances, resample);
            }
        } else {
            System.err.println("Pre-computing dependencies on the full dataset...");
        }

        System.err.println("Using " + dataForComputation.numInstances() + " instances for dependency calculations.");
        Measure dependencyCalculator = new WeakGamma();
        
        if (!m_unsupervised) {
            setupFuzzyMeasure(dependencyCalculator, dataForComputation, m_classIndex);
            for (int i = 0; i < m_numAttribs; i++) {
                if (i == m_classIndex) continue;
                BitSet tempSubset = new BitSet(m_numAttribs);
                tempSubset.set(i);
                m_featureClassDependencies[i] = dependencyCalculator.calculate(tempSubset);
            }
        }

        for (int i = 0; i < m_numAttribs; i++) {
            if (i == m_classIndex) continue;
            for (int j = i; j < m_numAttribs; j++) {
                if (j == m_classIndex || i == j) continue;
                
                Instances tempInstances = new Instances(dataForComputation);
                tempInstances.setClassIndex(j);
                
                Measure ffCalculator = new WeakGamma();
                setupFuzzyMeasure(ffCalculator, tempInstances, j);
                
                BitSet tempSubset = new BitSet(m_numAttribs);
                tempSubset.set(i);

                double dep = ffCalculator.calculate(tempSubset);
                m_featureFeatureDependencies[i][j] = dep;
                m_featureFeatureDependencies[j][i] = dep;
            }
        }
        System.err.println("Dependency pre-computation complete.");
    }
    
    private void setupFuzzyMeasure(Measure measure, Instances instances, int classIdx) throws Exception {
        m_Similarity.setInstances(instances);
        m_SimilarityEq.setInstances(instances);
        
        Similarity decisionSim;
        if (classIdx >= 0 && instances.attribute(classIdx).isNumeric()) {
            decisionSim = m_Similarity;
        } else {
            decisionSim = m_SimilarityEq;
        }
        decisionSim.setInstances(instances);

        TNorm composition = m_Similarity.getTNorm();
        measure.set(m_Similarity, decisionSim, m_TNorm, composition, m_Implicator, m_SNorm,
                instances.numInstances(), instances.numAttributes(), classIdx, instances);
    }
    
    @Override
    public double evaluateSubset(BitSet subset) throws Exception {
        int k = subset.cardinality();

        if (k == 0) {
            return 0.0;
        }

        if (m_unsupervised) {
        	return 0.0;
        }
        
        double sumFeatureClassDep = 0.0;
        for (int i = subset.nextSetBit(0); i >= 0; i = subset.nextSetBit(i + 1)) {
            sumFeatureClassDep += m_featureClassDependencies[i];
        }

        double avgFeatureClassDep = sumFeatureClassDep / k;

        double sumFeatureFeatureDep = 0.0;
        if (k > 1) {
            for (int i = subset.nextSetBit(0); i >= 0; i = subset.nextSetBit(i + 1)) {
                for (int j = subset.nextSetBit(i + 1); j >= 0; j = subset.nextSetBit(j + 1)) {
                    sumFeatureFeatureDep += m_featureFeatureDependencies[i][j];
                }
            }
        }
        
        double avgFeatureFeatureDep = (k > 1) ? sumFeatureFeatureDep / (k * (k - 1) / 2.0) : 0;
        
        double merit = (k * avgFeatureClassDep) / Math.sqrt(k + k * (k - 1) * avgFeatureFeatureDep);

        if (Double.isNaN(merit) || Double.isInfinite(merit)) {
            return 0.0;
        }

        return merit;
    }
    
    protected void resetOptions() {
        m_trainInstances = null;
        m_Similarity = new Similarity3();
        setTNorm(new TNormLukasiewicz());
        m_Implicator = new ImplicatorLukasiewicz();
        m_samplePercentage = 100.0;
        m_unsupervised = false;
        try {
            m_Similarity.setTNorm(new TNormLukasiewicz());
        } catch (Exception e) {
            // Should not happen with default
        }
    }

    public static void main(String[] args) {
        runEvaluator(new FuzzyRoughCFSEval(), args);
    }
}