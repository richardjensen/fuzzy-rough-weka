package weka.attributeSelection;

import weka.attributeSelection.ann.*;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.ivmeasure.IVFuzzyMeasure;
import weka.fuzzy.measure.*;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity3;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormLukasiewicz;

import java.util.*;
import java.util.function.Supplier;

public class FuzzyRoughSubsetEval
        extends ASEvaluation
        implements SubsetEvaluator, OptionHandler, TechnicalInformationHandler {

    static final long serialVersionUID = 747878400839276317L;

    private Instances m_trainInstances;
    private int m_classIndex;
    private int m_numAttribs;
    private int m_numInstances;
    private boolean m_isNumeric;
    public boolean computeCore = false;
    BitSet core;
    protected double c_divisor = -1;
    public Similarity m_Similarity = new Similarity3();
    public Similarity m_SimilarityEq = new SimilarityEq();
    public Similarity m_DecisionSimilarity = new SimilarityEq();
    protected Measure m_Measure = new WeakGamma();
    public TNorm m_TNorm = new TNormLukasiewicz();
    public TNorm m_composition = new TNormLukasiewicz();
    public Implicator m_Implicator = new ImplicatorLukasiewicz();
    public SNorm m_SNorm = new SNormLukasiewicz();
    public boolean unsupervised = false;

    // --- ANN related members ---
    public enum ANNAproach {
        NONE, LSH, ANNOY_RP, ANNOY_TP, NSW, HNSW
    }

    protected ANNAproach m_annApproach = ANNAproach.NONE;
    protected int m_lshBands = 10;
    protected int m_lshRows = 2;
    protected int m_annoyNumTrees = 10;
    protected int m_nswM = 15;
    protected int m_hnswM = 16;
    protected int m_hnswEfConstruction = 100;

    public FuzzyRoughSubsetEval() {
        resetOptions();
    }
    

    
    @Override
    public TechnicalInformation getTechnicalInformation() {
        // ... (Technical information as provided before) ...
        return new TechnicalInformation(Type.ARTICLE); // Placeholder
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<>();
        
        newVector.addElement(new Option("\tSimilarity relation.", "R", 1, "-R <full.class.name> [options]"));
        newVector.addElement(new Option("\tMeasure to use.", "Z", 1, "-Z <full.class.name> [options]"));
        newVector.addElement(new Option("\tImplicator to use.", "I", 1, "-I <full.class.name> [options]"));
        newVector.addElement(new Option("\tT-Norm to use.", "T", 1, "-T <full.class.name> [options]"));
        newVector.addElement(new Option("\tStart with the core.", "L", 0, "-L"));
        newVector.addElement(new Option("\tUnsupervised mode.", "U", 0, "-U"));

        newVector.addElement(new Option("\tANN approach for nearest enemies.", "ANN", 1, "-ANN <NONE | LSH | ANNOY_RP | ANNOY_TP | NSW | HNSW>"));
        newVector.addElement(new Option("\tLSH: Number of bands (b).", "ANN-lsh-b", 1, "-ANN-lsh-b <num>"));
        newVector.addElement(new Option("\tLSH: Number of rows per band (r).", "ANN-lsh-r", 1, "-ANN-lsh-r <num>"));
        newVector.addElement(new Option("\tAnnoy: Number of trees.", "ANN-annoy-t", 1, "-ANN-annoy-t <num>"));
        newVector.addElement(new Option("\tNSW: Neighbors per node (M).", "ANN-nsw-m", 1, "-ANN-nsw-m <num>"));
        newVector.addElement(new Option("\tHNSW: Neighbors per node (M).", "ANN-hnsw-m", 1, "-ANN-hnsw-m <num>"));
        newVector.addElement(new Option("\tHNSW: Build-time search depth (efConstruction).", "ANN-hnsw-efc", 1, "-ANN-hnsw-efc <num>"));

        return newVector.elements();
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        resetOptions();
        String optionString;

        optionString = Utils.getOption('Z', options);
        if (optionString.length() != 0) {
            String[] spec = Utils.splitOptions(optionString);
            setMeasure((Measure) Utils.forName(Measure.class, spec[0], Arrays.copyOfRange(spec, 1, spec.length)));
        }
        optionString = Utils.getOption('I', options);
        if (optionString.length() != 0) {
            String[] spec = Utils.splitOptions(optionString);
            setImplicator((Implicator) Utils.forName(Implicator.class, spec[0], Arrays.copyOfRange(spec, 1, spec.length)));
        }
        optionString = Utils.getOption('T', options);
        if (optionString.length() != 0) {
            String[] spec = Utils.splitOptions(optionString);
            setTNorm((TNorm) Utils.forName(TNorm.class, spec[0], Arrays.copyOfRange(spec, 1, spec.length)));
        }
        optionString = Utils.getOption('R', options);
        if (optionString.length() != 0) {
            String[] spec = Utils.splitOptions(optionString);
            setSimilarity((Similarity) Utils.forName(Similarity.class, spec[0], Arrays.copyOfRange(spec, 1, spec.length)));
        }

        setComputeCore(Utils.getFlag('L', options));
        setUnsupervised(Utils.getFlag('U', options));

        String annString = Utils.getOption("ANN", options);
        if (annString.length() != 0) setAnnApproach(ANNAproach.valueOf(annString.toUpperCase()));
        String lshBandsStr = Utils.getOption("ANN-lsh-b", options);
        if (lshBandsStr.length() != 0) setLshBands(Integer.parseInt(lshBandsStr));
        String lshRowsStr = Utils.getOption("ANN-lsh-r", options);
        if (lshRowsStr.length() != 0) setLshRows(Integer.parseInt(lshRowsStr));
        String annoyTreesStr = Utils.getOption("ANN-annoy-t", options);
        if (annoyTreesStr.length() != 0) setAnnoyNumTrees(Integer.parseInt(annoyTreesStr));
        String nswMStr = Utils.getOption("ANN-nsw-m", options);
        if (nswMStr.length() != 0) setNswM(Integer.parseInt(nswMStr));
        String hnswMStr = Utils.getOption("ANN-hnsw-m", options);
        if (hnswMStr.length() != 0) setHnswM(Integer.parseInt(hnswMStr));
        String hnswEfcStr = Utils.getOption("ANN-hnsw-efc", options);
        if (hnswEfcStr.length() != 0) setHnswEfConstruction(Integer.parseInt(hnswEfcStr));
    }

    @Override
    public String[] getOptions() {
        Vector<String> result = new Vector<>();
        java.util.function.Function<String[], String> safeJoin = (opts) ->
                (opts == null || opts.length == 0) ? "" : Utils.joinOptions(opts);

        if (m_Measure != null) {
            result.add("-Z");
            result.add((m_Measure.getClass().getName() + " " + safeJoin.apply(m_Measure.getOptions())).trim());
        }
        if (m_Implicator != null) {
            result.add("-I");
            result.add((m_Implicator.getClass().getName() + " " + safeJoin.apply(m_Implicator.getOptions())).trim());
        }
        if (m_TNorm != null) {
            result.add("-T");
            result.add((m_TNorm.getClass().getName() + " " + safeJoin.apply(m_TNorm.getOptions())).trim());
        }
        if (m_Similarity != null) {
            result.add("-R");
            result.add((m_Similarity.getClass().getName() + " " + safeJoin.apply(m_Similarity.getOptions())).trim());
        }
        if (getComputeCore()) result.add("-L");
        if (getUnsupervised()) result.add("-U");

        result.add("-ANN"); result.add(m_annApproach.name());
        result.add("-ANN-lsh-b"); result.add("" + getLshBands());
        result.add("-ANN-lsh-r"); result.add("" + getLshRows());
        result.add("-ANN-annoy-t"); result.add("" + getAnnoyNumTrees());
        result.add("-ANN-nsw-m"); result.add("" + getNswM());
        result.add("-ANN-hnsw-m"); result.add("" + getHnswM());
        result.add("-ANN-hnsw-efc"); result.add("" + getHnswEfConstruction());

        return result.toArray(new String[0]);
    }
    
    @Override
    public void buildEvaluator(Instances data) throws Exception {
        getCapabilities().testWithFail(data);

        m_trainInstances = new Instances(data);
        m_numAttribs = m_trainInstances.numAttributes();
        m_numInstances = m_trainInstances.numInstances();
        m_classIndex = m_trainInstances.classIndex();

        if (m_classIndex < 0 || unsupervised) {
            unsupervised = true;
            m_classIndex = -1;
        } else {
            m_isNumeric = m_trainInstances.attribute(m_classIndex).isNumeric();
            if (m_isNumeric) m_DecisionSimilarity = m_Similarity;
            else m_DecisionSimilarity = new SimilarityEq();
        }

        m_Similarity.setInstances(m_trainInstances);
        m_DecisionSimilarity.setInstances(m_trainInstances);
        m_SimilarityEq.setInstances(m_trainInstances);
        m_composition = m_Similarity.getTNorm();
        m_Measure.set(m_Similarity, m_DecisionSimilarity, m_TNorm, m_composition, m_Implicator, m_SNorm, m_numInstances, m_numAttribs, m_classIndex, m_trainInstances);

        if (m_annApproach != ANNAproach.NONE) {
            System.err.println("Building ANN index for OVR search: " + m_annApproach);
            long startTime = System.currentTimeMillis();

            int numFeatures = m_trainInstances.numAttributes() - (m_trainInstances.classIndex() >= 0 ? 1 : 0);

            OvrAnnSearcher searcher = new OvrAnnSearcher(() -> {
                switch (m_annApproach) {
                    case LSH: return new LSH(m_lshBands, m_lshRows, numFeatures);
                    case ANNOY_RP: return new AnnoyRandomProjection(m_annoyNumTrees, numFeatures);
                    case ANNOY_TP: return new AnnoyTwoPointSplit(m_annoyNumTrees, numFeatures);
                    case NSW:
                        return new NavigableSmallWorld(m_nswM, AnnUtils.createFrfsDistanceFunction(m_trainInstances, m_Similarity));
                    case HNSW:
                        return new HNSW(m_hnswM, m_hnswEfConstruction, AnnUtils.createFrfsDistanceFunction(m_trainInstances, m_Similarity));
                    default: throw new IllegalStateException("Unknown ANN Approach");
                }
            });

            searcher.buildIndex(m_trainInstances);
            System.err.println(" -> Index built in " + (System.currentTimeMillis() - startTime) + " ms.");

            System.err.println("Pre-computing nearest enemy cache...");
            startTime = System.currentTimeMillis();
            Map<Integer, int[]> neighborCache = new HashMap<>();
            for (int i = 0; i < m_numInstances; i++) {
                neighborCache.put(i, searcher.queryEnemies(m_trainInstances.instance(i)));
            }
            m_Measure.setNeighborCache(neighborCache);
            System.err.println(" -> Cache built in " + (System.currentTimeMillis() - startTime) + " ms.");
        }
        if (c_divisor == -1) c_divisor = getConsistency();
    }
    
    @Override
    public double evaluateSubset(BitSet subset) throws Exception {
        double eval = 0.0;
        if (subset.cardinality() > 0) {
            eval = m_Measure.calculate(subset);
            if (!((m_Measure instanceof FuzzyMutInf) || (m_Measure instanceof RoughMutInf))) {
                if (c_divisor != -1) eval = Math.min(1, eval / c_divisor);
            }
        }
        return eval;
    }

    // --- Getters, Setters, and Tip Texts for all options ---

    public void setMeasure(Measure fe) { m_Measure = fe; }
    public Measure getMeasure() { return m_Measure; }
    public void setUnsupervised(boolean u) { unsupervised = u; }
    public boolean getUnsupervised() { return unsupervised; }
    public String computeCoreTipText() { return "Only works with appropriate search methods: HillClimber, PSO, AntSearch, GeneticSearch"; }
    public void setComputeCore(boolean flag) { computeCore = flag; }
    public boolean getComputeCore() { return computeCore; }
    public void setImplicator(Implicator impl) { m_Implicator = impl; }
    public Implicator getImplicator() { return m_Implicator; }
    public void setTNorm(TNorm tnorm) {
        m_TNorm = tnorm;
        m_SNorm = tnorm.getAssociatedSNorm();
    }
    public TNorm getTNorm() { return m_TNorm; }
    public Similarity getSimilarity() { return m_Similarity; }
    public void setSimilarity(Similarity s) { m_Similarity = s; }

    public void setAnnApproach(ANNAproach approach) { m_annApproach = approach; }
    public ANNAproach getAnnApproach() { return m_annApproach; }
    public String annApproachTipText() { return "Approximate Nearest Neighbor method for finding nearest enemies."; }

    public void setLshBands(int b) { m_lshBands = b; }
    public int getLshBands() { return m_lshBands; }
    public String lshBandsTipText() { return "Number of bands for LSH."; }

    public void setLshRows(int r) { m_lshRows = r; }
    public int getLshRows() { return m_lshRows; }
    public String lshRowsTipText() { return "Number of rows per band for LSH."; }
    
    public void setAnnoyNumTrees(int t) { m_annoyNumTrees = t; }
    public int getAnnoyNumTrees() { return m_annoyNumTrees; }
    public String annoyNumTreesTipText() { return "Number of trees for Annoy."; }

    public void setNswM(int m) { m_nswM = m; }
    public int getNswM() { return m_nswM; }
    public String nswMTipText() { return "Number of neighbors per node for NSW."; }

    public void setHnswM(int m) { m_hnswM = m; }
    public int getHnswM() { return m_hnswM; }
    public String hnswMTipText() { return "Number of neighbors per node for HNSW."; }

    public void setHnswEfConstruction(int efc) { m_hnswEfConstruction = efc; }
    public int getHnswEfConstruction() { return m_hnswEfConstruction; }
    public String hnswEfConstructionTipText() { return "Build-time search depth for HNSW."; }
    
    // --- Other Methods ---

    private double getConsistency() throws Exception {
        BitSet full = new BitSet(m_numAttribs);
        for (int a = 0; a < m_numAttribs; a++) if (a != m_classIndex) full.set(a);
        
        double c_div = m_Measure.calculate(full);
        if ((m_Measure instanceof FuzzyMutInf) || (m_Measure instanceof RoughMutInf)) return 1.0;
        if (c_div == 0 || Double.isNaN(c_div)) {
            System.err.println("\n*** Inconsistent data (full dataset value = " + c_div + " for this measure) ***\n");
            return 1.0;
        }
        return c_div;
    }

    public BitSet computeCore() throws Exception {
        BitSet full = new BitSet(m_numAttribs);
        for (int a = 0; a < m_numAttribs; a++) full.set(a);

        FuzzyMeasure gammaEvaluator = m_isNumeric ? new Gamma() : new WeakGamma();
        gammaEvaluator.set(m_Similarity, m_DecisionSimilarity, m_TNorm, m_composition, m_Implicator, m_SNorm, m_numInstances, m_numAttribs, m_classIndex, m_trainInstances);
        double gammaFull = gammaEvaluator.calculate(full);

        System.err.println("Calculating core attributes... Full dataset dependency is: " + gammaFull);
        System.err.print("Core: [ ");

        BitSet core1 = new BitSet(m_numAttribs);
        for (int a = 0; a < m_numAttribs; a++) {
            if (a == m_classIndex) continue;
            full.clear(a);
            double gamma = gammaEvaluator.calculate(full);
            full.set(a);
            if (gamma < gammaFull) {
                core1.set(a);
                System.err.print((a + 1) + " ");
            }
        }
        System.err.println("]");
        this.core = core1;
        return core1;
    }

    protected void resetOptions() {
        m_trainInstances = null;
        m_Measure = new WeakGamma();
        m_Implicator = new ImplicatorLukasiewicz();
        m_TNorm = new TNormLukasiewicz();
        m_Similarity = new Similarity3();
        try {
            m_Similarity.setTNorm(new TNormLukasiewicz());
        } catch (Exception e) {
            // Should not happen with default
        }
        m_annApproach = ANNAproach.NONE;
        // reset other params to defaults
        m_lshBands = 10;
        m_lshRows = 2;
        m_annoyNumTrees = 10;
        m_nswM = 15;
        m_hnswM = 16;
        m_hnswEfConstruction = 100;
    }

    @Override
    public String toString() {
        StringBuilder text = new StringBuilder();
        if (m_trainInstances == null) {
            text.append("FRFS feature evaluator has not been built yet\n");
        } else {
            text.append("\nFuzzy rough feature selection\n\nMethod: ").append(m_Measure);
            if (m_Measure instanceof IVFuzzyMeasure) {
                text.append("\nParameter = ").append(((IVFuzzyMeasure) m_Measure).getParameter()).append("\n");
            }
            text.append("\nSimilarity measure: ").append(m_Similarity);
            text.append("\nDecision similarity: ").append(m_DecisionSimilarity);
            text.append("\nImplicator: ").append(m_Implicator);
            text.append("\nT-Norm: ").append(m_TNorm);
            if (m_Similarity != null && m_Similarity.getTNorm() != null) {
                text.append("\nRelation composition: ").append(m_Similarity.getTNorm());
            }
            text.append("\n(S-Norm: ").append(m_SNorm).append(")\n\n");
            text.append("ANN Acceleration: ").append(m_annApproach).append("\n");
            text.append("Dataset consistency: ").append(c_divisor).append("\n");
            if (computeCore && core != null) {
                StringBuilder coreString = new StringBuilder(" ");
                for (int a = core.nextSetBit(0); a >= 0; a = core.nextSetBit(a + 1)) {
                    coreString.append(a + 1).append(" ");
                }
                text.append("Core: {").append(coreString.toString().trim()).append("}\n\n");
            }
        }
        return text.toString();
    }
    
    public static void main(String[] args) {
        runEvaluator(new FuzzyRoughSubsetEval(), args);
    }

    /**
     * Calculates and outputs the inverse granularity for each attribute.
     * @return An array of strings representing the rounded inverse granularity of each attribute.
     */
    public String[] outputGranularities() {
        if (m_trainInstances == null || m_Measure == null) {
            System.err.println("Evaluator has not been built yet. Cannot calculate granularities.");
            return new String[0]; 
        }

        String[] output = new String[m_numAttribs];
        
        if (!(m_Measure instanceof FuzzyMeasure)) {
            for (int a = 0; a < m_numAttribs; a++) {
                output[a] = (a == m_classIndex) ? "class" : "N/A";
            }
            System.err.println("Warning: The selected measure '" + m_Measure.getClass().getSimpleName() + "' does not support granularity calculation.");
            return output;
        }
        
        FuzzyMeasure fm = (FuzzyMeasure) m_Measure;
        
        for (int a = 0; a < m_numAttribs; a++) {
            if (a != m_classIndex) {
                double granularity = fm.granularity(a);
                
                // Handle case where granularity is zero to avoid division by zero -> Infinity
                if (granularity == 0.0) {
                     output[a] = "Infinity";
                } else {
                    double invGranularity = Math.round(1.0 / granularity);
                    output[a] = "" + (long)invGranularity;
                }
            } else {
                output[a] = "class";
            }
        }
        return output;
    }
}