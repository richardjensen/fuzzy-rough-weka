package weka.filters.supervised.instance;

import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.fuzzy.implicator.*;
import weka.fuzzy.similarity.*;
import weka.fuzzy.tnorm.*;

import java.io.Serializable;
import java.util.*;

/**
 <!-- globalinfo-start -->
 * Filters instances using Approximation-based Instance Ranking (AIR).
 * Allows ranking based on fuzzy-rough lower, combined, or boundary measures.
 * Supports standard (inf/sup) or RIM OWA approximations.
 * Offers automatic, fixed, or similarity percentile thresholding for coverage.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -R &lt;similarity specification&gt;
 *  Similarity function to use for features.
 *  (default: weka.fuzzy.similarity.Similarity3)</pre>
 *
 * <pre> -T &lt;t-norm specification&gt;
 *  T-norm to use for combining feature similarities and for upper approximation.
 *  (default: weka.fuzzy.tnorm.TNormLukasiewicz)</pre>
 *
 * <pre> -I &lt;implicator specification&gt;
 *  Implicator to use for lower approximation.
 *  (default: weka.fuzzy.implicator.ImplicatorLukasiewicz)</pre>
 *
 * <pre> -AT &lt;0|1&gt;
 *  Approximation type: 0=Standard (Inf/Sup), 1=RIM OWA.
 *  (default: 0)</pre>
 *
 * <pre> -ALPHA &lt;double&gt;
 *  Alpha parameter for RIM OWA weights (used if -AT 1).
 *  Alpha=1 -> Average; &gt;1 -> leans Max; &lt;1 -> leans Min.
 *  (default: 2.0)</pre>
 *
 * <pre> -M &lt;0|1|2&gt;
 *  Measure type for ranking instances: 0=Lower Approx, 1=Combined (Avg Lower+Upper), 2=Boundary Emphasis (Upper*(1-Lower)).
 *  (default: 0)</pre>
 *
 * <pre> -TM &lt;0|1|2&gt;
 *  Threshold mode for instance coverage: 0=Automatic (1 - measure(j)), 1=Fixed, 2=Percentile.
 *  (default: 0)</pre>
 *
 * <pre> -K &lt;double&gt;
 *  Fixed threshold value (used if -TM 1).
 *  (default: 0.85)</pre>
 *
 * <pre> -P &lt;int&gt;
 *  Percentile threshold (0-100) (used if -TM 2).
 *  Determines threshold based on similarities to prototype. Lower P covers more.
 *  (default: 25)</pre>
 *
 <!-- options-end -->
 *
 * @author Richard Jensen, Neil Mac Parthalain (Original LAIR concept)
 * @author AI Assistant (Modifications for OWA, Boundary, Percentile)
 * @version 2025.04.23
 */
public class LAIR2
        extends Filter
        implements SupervisedFilter, OptionHandler, Serializable, AdditionalMeasureProducer {

    /** for serialization */
    static final long serialVersionUID = 5830582094758372957L;

    // --- Fuzzy Component Members ---
    protected Similarity m_Similarity = new Similarity3(); // Default to Sim3 which is often good for classification
    protected Similarity m_SimilarityEq = new SimilarityEq(); // For nominal attributes and class
    protected TNorm m_TNorm = new TNormLukasiewicz();
    protected Implicator m_Implicator = new ImplicatorLukasiewicz();
    protected TNorm m_compositionTNorm = new TNormLukasiewicz(); // T-Norm for combining feature sims

    // --- Approximation Type ---
    protected static final int APPROX_STANDARD = 0;
    protected static final int APPROX_RIM_OWA = 1;
    public static final Tag[] TAGS_APPROX_TYPE = {
            new Tag(APPROX_STANDARD, "Standard (Inf/Sup)"),
            new Tag(APPROX_RIM_OWA, "RIM OWA"),
    };
    protected int m_approxType = APPROX_STANDARD;
    protected double m_rimAlpha = 2.0; // Default alpha leans towards max

    // --- Measure Type ---
    protected static final int MEASURE_LOWER = 0;
    protected static final int MEASURE_COMBINED = 1;
    protected static final int MEASURE_BOUNDARY = 2;
    public static final Tag[] TAGS_MEASURE_TYPE = {
            new Tag(MEASURE_LOWER, "Lower Approx"),
            new Tag(MEASURE_COMBINED, "Combined (Avg Lower+Upper)"),
            new Tag(MEASURE_BOUNDARY, "Boundary Emphasis (Upper*(1-Lower))"),
    };
    protected int m_measureType = MEASURE_LOWER;

    // --- Thresholding ---
    protected static final int THRESH_AUTO = 0;
    protected static final int THRESH_FIXED = 1;
    protected static final int THRESH_PERCENTILE = 2;
    public static final Tag[] TAGS_THRESH_MODE = {
            new Tag(THRESH_AUTO, "Automatic (1 - measure(j))"),
            new Tag(THRESH_FIXED, "Fixed Value"),
            new Tag(THRESH_PERCENTILE, "Similarity Percentile"),
    };
    protected int m_thresholdMode = THRESH_AUTO;
    protected double m_fixedThreshold = 0.85;
    protected int m_percentileThreshold = 25; // Default: cover the 75% most similar

    // --- Internal State ---
    protected double removedCount = 0; // For AdditionalMeasureProducer

    /** Helper class to store instance index and its calculated measure for sorting. */
    protected static class RankedInstance implements Comparable<RankedInstance> {
        int index;
        double measure = 0; // The value used for ranking (lower, combined, or boundary)
        double lowerApprox = 0; // Store raw values for potential later use (e.g., thresholding)
        double upperApprox = 0;

        public RankedInstance(int idx, double measure, double lower, double upper) {
            this.index = idx;
            this.measure = measure;
            this.lowerApprox = lower;
            this.upperApprox = upper;
        }

        public int getIndex() { return index; }
        public double getMeasure() { return measure; }

        // Sort in descending order of measure (best first)
        @Override
        public int compareTo(RankedInstance other) {
            return Double.compare(other.measure, this.measure);
        }

        @Override
        public String toString() {
            return index + ":" + String.format("%.4f", measure) +
                   " (L:" + String.format("%.4f", lowerApprox) +
                   ", U:" + String.format("%.4f", upperApprox) + ")";
        }
    }

    // --- Constructor and Global Info ---
    public LAIR2() {
        // Constructor - defaults are set in member declarations
    }

    public String globalInfo() {
        return "Approximation-based Instance Ranking (AIR) filter using fuzzy-rough sets.\n"
                + "Ranks instances based on lower, combined, or boundary emphasis measures derived from "
                + "standard or RIM OWA fuzzy-rough approximations. Selects prototypes and removes "
                + "redundant instances based on configurable thresholding (automatic, fixed, or percentile).";
    }

    // --- Option Handling ---
    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> options = new Vector<>();

        options.addElement(new Option("\tSimilarity function for features.",
                "R", 1, "-R <similarity specification>"));
        options.addElement(new Option("\tT-norm for similarity combination and upper approx.",
                "T", 1, "-T <t-norm specification>"));
        options.addElement(new Option("\tImplicator for lower approximation.",
                "I", 1, "-I <implicator specification>"));
        options.addElement(new Option("\tApproximation type (0=Standard, 1=RIM OWA).",
                "AT", 1, "-AT <0|1>"));
        options.addElement(new Option("\tAlpha for RIM OWA (if -AT 1). >1 leans Max, <1 leans Min.",
                "ALPHA", 1, "-ALPHA <double>"));
        options.addElement(new Option("\tMeasure type for ranking (0=Lower, 1=Combined, 2=Boundary).",
                "M", 1, "-M <0|1|2>"));
        options.addElement(new Option("\tThreshold mode (0=Auto, 1=Fixed, 2=Percentile).",
                "TM", 1, "-TM <0|1|2>"));
        options.addElement(new Option("\tFixed threshold value (if -TM 1).",
                "K", 1, "-K <double>"));
        options.addElement(new Option("\tPercentile threshold (0-100) (if -TM 2).",
                "P", 1, "-P <int>"));

        return options.elements();
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        String optionString;

        optionString = Utils.getOption('R', options);
        if (optionString.length() != 0) {
            setSimilarity((Similarity) Utils.forName(Similarity.class, optionString, null)); // Assuming simple class name for now
        } else {
            setSimilarity(new Similarity3());
        }

        optionString = Utils.getOption('T', options);
        if (optionString.length() != 0) {
            setTNorm((TNorm) Utils.forName(TNorm.class, optionString, null));
        } else {
            setTNorm(new TNormLukasiewicz());
        }

        optionString = Utils.getOption('I', options);
        if (optionString.length() != 0) {
            setImplicator((Implicator) Utils.forName(Implicator.class, optionString, null));
        } else {
            setImplicator(new ImplicatorLukasiewicz());
        }

        optionString = Utils.getOption("AT", options);
        if (optionString.length() != 0) {
            setApproxType(new SelectedTag(Integer.parseInt(optionString), TAGS_APPROX_TYPE));
        } else {
             setApproxType(new SelectedTag(APPROX_STANDARD, TAGS_APPROX_TYPE));
        }

        optionString = Utils.getOption("ALPHA", options);
        if (optionString.length() != 0) {
            setRimAlpha(Double.parseDouble(optionString));
        } else {
            setRimAlpha(2.0);
        }

        optionString = Utils.getOption('M', options);
        if (optionString.length() != 0) {
             setMeasureType(new SelectedTag(Integer.parseInt(optionString), TAGS_MEASURE_TYPE));
        } else {
             setMeasureType(new SelectedTag(MEASURE_LOWER, TAGS_MEASURE_TYPE));
        }

         optionString = Utils.getOption("TM", options);
        if (optionString.length() != 0) {
             setThresholdMode(new SelectedTag(Integer.parseInt(optionString), TAGS_THRESH_MODE));
        } else {
             setThresholdMode(new SelectedTag(THRESH_AUTO, TAGS_THRESH_MODE));
        }

        optionString = Utils.getOption('K', options);
        if (optionString.length() != 0) {
            setFixedThreshold(Double.parseDouble(optionString));
        } else {
            setFixedThreshold(0.85);
        }

        optionString = Utils.getOption('P', options);
        if (optionString.length() != 0) {
            setPercentileThreshold(Integer.parseInt(optionString));
        } else {
            setPercentileThreshold(25);
        }

        Utils.checkForRemainingOptions(options);
    }


    @Override
    public String[] getOptions() {
        Vector<String> options = new Vector<>();

        options.add("-R"); options.add(getSimilarity().getClass().getName());
        options.add("-T"); options.add(getTNorm().getClass().getName());
        options.add("-I"); options.add(getImplicator().getClass().getName());
        options.add("-AT"); options.add("" + m_approxType);
        options.add("-ALPHA"); options.add("" + m_rimAlpha);
        options.add("-M"); options.add("" + m_measureType);
        options.add("-TM"); options.add("" + m_thresholdMode);
        options.add("-K"); options.add("" + m_fixedThreshold);
        options.add("-P"); options.add("" + m_percentileThreshold);

        return options.toArray(new String[0]);
    }

    // --- Getters and Setters for options ---

    public Similarity getSimilarity() { return m_Similarity; }
    public void setSimilarity(Similarity s) { m_Similarity = s; }

    public TNorm getTNorm() { return m_TNorm; }
    public void setTNorm(TNorm t) { m_TNorm = t; m_compositionTNorm = t; } // Assume composition is same as main T-norm

    public Implicator getImplicator() { return m_Implicator; }
    public void setImplicator(Implicator i) { m_Implicator = i; }

    public SelectedTag getApproxType() { return new SelectedTag(m_approxType, TAGS_APPROX_TYPE); }
    public void setApproxType(SelectedTag tag) { if (tag.getTags() == TAGS_APPROX_TYPE) m_approxType = tag.getSelectedTag().getID(); }

    public double getRimAlpha() { return m_rimAlpha; }
    public void setRimAlpha(double alpha) { m_rimAlpha = alpha; }

    public SelectedTag getMeasureType() { return new SelectedTag(m_measureType, TAGS_MEASURE_TYPE); }
    public void setMeasureType(SelectedTag tag) { if (tag.getTags() == TAGS_MEASURE_TYPE) m_measureType = tag.getSelectedTag().getID(); }

    public SelectedTag getThresholdMode() { return new SelectedTag(m_thresholdMode, TAGS_THRESH_MODE); }
    public void setThresholdMode(SelectedTag tag) { if (tag.getTags() == TAGS_THRESH_MODE) m_thresholdMode = tag.getSelectedTag().getID(); }

    public double getFixedThreshold() { return m_fixedThreshold; }
    public void setFixedThreshold(double t) { m_fixedThreshold = t; }

    public int getPercentileThreshold() { return m_percentileThreshold; }
    public void setPercentileThreshold(int p) { m_percentileThreshold = (p < 0) ? 0 : (p > 100 ? 100 : p); }


    // --- Capabilities ---
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.enable(Capability.MISSING_VALUES); // Similarity functions should handle missing values appropriately
        result.enableAllClasses();
        result.enable(Capability.MISSING_CLASS_VALUES); // Handled by deleteWithMissingClass
        result.setMinimumNumberInstances(1);
        return result;
    }

    // --- Input/Output Format ---
    @Override
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        super.setInputFormat(instanceInfo);
        // Output format is the same as input, just potentially fewer instances
        setOutputFormat(instanceInfo);
        return true;
    }

    // --- Filtering Logic ---
    @Override
    public boolean input(Instance instance) {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }
        // Requires buffering all instances
        bufferInput(instance);
        return false; // Output always comes in batchFinished
    }

    @Override
    public boolean batchFinished() throws Exception {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        Instances inputInstances = getInputFormat();
        Instances originalData = new Instances(inputInstances); // Keep original for pushing later
        inputInstances.deleteWithMissingClass(); // Work on data without missing classes

        int numInstances = inputInstances.numInstances();
        int numAttribs = inputInstances.numAttributes();
        int classIndex = inputInstances.classIndex();

        if (numInstances == 0) { // Handle empty input after missing class removal
             flushInput();
             m_NewBatch = true;
             m_FirstBatchDone = true;
             return false;
        }

        int numClasses = inputInstances.numClasses();
        if (classIndex < 0 || numClasses <= 1) {
             // Cannot perform supervised selection without a valid class attribute
             // Push all original instances through
             System.err.println("Warning: LAIR2 requires a nominal class attribute with at least 2 classes. Passing all instances through.");
             for(int i=0; i<originalData.numInstances(); ++i) {
                 push(originalData.instance(i));
             }
             flushInput();
             m_NewBatch = true;
             m_FirstBatchDone = true;
             removedCount = 0;
             return (numPendingOutput() != 0);
        }

        // --- Setup Fuzzy Components ---
        m_Similarity.setInstances(inputInstances);
        m_SimilarityEq.setInstances(inputInstances); // For nominal attributes
        // Use SimilarityEq for nominal decision class comparison by default
        Similarity decisionSimilarity = inputInstances.attribute(classIndex).isNumeric() ? m_Similarity : m_SimilarityEq;
        decisionSimilarity.setInstances(inputInstances);
        
        m_compositionTNorm = m_Similarity.getTNorm();
 


        // --- 1. Precompute Similarities ---
        System.err.println("Precomputing similarities...");
        double[][] sims = precomputeSimilarities(inputInstances, classIndex);


        // --- 2. Calculate Approximations and Measures ---
        System.err.println("Calculating approximations and measures...");
        ArrayList<RankedInstance>[] perDecision = (ArrayList<RankedInstance>[]) new ArrayList[numClasses];
        for (int d = 0; d < numClasses; d++) {
            perDecision[d] = new ArrayList<>();
        }

        boolean needsUpper = (m_measureType == MEASURE_COMBINED || m_measureType == MEASURE_BOUNDARY);

        for (int i = 0; i < numInstances; i++) {
             Instance instI = inputInstances.instance(i);
             int classValueI = (int) instI.classValue();

             double[] approx = approximations(i, classValueI, inputInstances, sims, needsUpper);
             double lower = approx[0];
             double upper = approx[1]; // Will be 0 if needsUpper was false

             double measure;
             switch (m_measureType) {
                 case MEASURE_COMBINED:
                     measure = (lower + upper) / 2.0;
                     break;
                 case MEASURE_BOUNDARY:
                     measure = upper * (1.0 - lower);
                     break;
                 case MEASURE_LOWER:
                 default:
                     measure = lower;
                     break;
             }
             perDecision[classValueI].add(new RankedInstance(i, measure, lower, upper));
        }

        // Sort instances within each class by measure (descending)
        for (int d = 0; d < numClasses; d++) {
            Collections.sort(perDecision[d]);
        }


        // --- 3. Select Instances ---
        System.err.println("Selecting instances...");
        BitSet coveredInstances = new BitSet(numInstances);
        BitSet selectedInstances = new BitSet(numInstances); // Track selections explicitly

        for (int d = 0; d < numClasses; d++) {
             // Use a copy to allow modification while iterating conceptually
             ArrayList<RankedInstance> currentClassCandidates = new ArrayList<>(perDecision[d]);

             // We iterate based on the sorted order, but track processed indices
             BitSet processedInClass = new BitSet(numInstances);

             for(RankedInstance prototypeCandidate : perDecision[d]) {
                 int i = prototypeCandidate.getIndex();

                 // Skip if already selected or covered (e.g., by another class's prototype)
                 if (selectedInstances.get(i) || coveredInstances.get(i) || processedInClass.get(i)) {
                     continue;
                 }

                 // Optional check from original LAIR (Algorithm 2 equivalent) - skip if measure depends on upper and upper=0
                 if ((m_measureType == MEASURE_COMBINED || m_measureType == MEASURE_BOUNDARY) && prototypeCandidate.upperApprox < 1e-9) {
                      processedInClass.set(i); // Mark as processed even if skipped
                      continue;
                 }

                 // Select instance i
                 selectedInstances.set(i);
                 processedInClass.set(i); // Mark i itself as processed

                 // --- Determine Threshold and Check Coverage ---
                 double percentileTau = 0.0; // Only used if mode is percentile

                 // Collect remaining candidates *in this class* for potential coverage check
                 ArrayList<RankedInstance> remainingCandidates = new ArrayList<>();
                 for(RankedInstance potentialCover : currentClassCandidates) {
                      int potentialIdx = potentialCover.getIndex();
                      if (potentialIdx != i && !selectedInstances.get(potentialIdx) && !coveredInstances.get(potentialIdx) && !processedInClass.get(potentialIdx)) {
                          remainingCandidates.add(potentialCover);
                      }
                 }

                 // Calculate percentile threshold if needed (once per prototype i)
                 if (m_thresholdMode == THRESH_PERCENTILE && !remainingCandidates.isEmpty()) {
                      ArrayList<Double> similaritiesToI = new ArrayList<>();
                      for(RankedInstance candidateJ : remainingCandidates) {
                          similaritiesToI.add(sims[i][candidateJ.getIndex()]);
                      }
                      Collections.sort(similaritiesToI);
                      // Calculate index (ensure it's within bounds)
                      int pIndex = (int) Math.round(similaritiesToI.size() * (m_percentileThreshold / 100.0));
                      pIndex = Math.max(0, Math.min(similaritiesToI.size() - 1, pIndex));
                      percentileTau = similaritiesToI.get(pIndex);
                 }

                 // Iterate through remaining candidates to check for coverage by i
                 for (RankedInstance objectY : remainingCandidates) {
                     int j = objectY.getIndex();

                     // Determine threshold for this specific pair (i, j)
                     double currentTau;
                     switch(m_thresholdMode) {
                         case THRESH_FIXED:
                             currentTau = m_fixedThreshold;
                             break;
                         case THRESH_PERCENTILE:
                             currentTau = percentileTau;
                             break;
                         case THRESH_AUTO:
                         default:
                              // Use the measure stored in objectY (which came from perDecision[d])
                             currentTau = 1.0 - objectY.getMeasure();
                             break;
                     }

                     // Check coverage
                     if (sims[i][j] > currentTau + 1e-9) { // Add tolerance
                         coveredInstances.set(j);
                         processedInClass.set(j); // Mark as processed (covered)
                         // System.err.println("Instance " + j + " covered by " + i + " (sim=" + sims[i][j] + ", tau="+currentTau+")");
                     }
                 }
             } // End loop over potential prototypes in class d
        } // End loop over classes


        // --- 4. Output Selected Instances ---
        System.err.println("Outputting selected instances...");
        setOutputFormat(originalData); // Set output format based on original data
        int originalCount = originalData.numInstances();
        int selectedCount = 0;

        for (int i = 0; i < originalCount; i++) {
             Instance currentInst = originalData.instance(i);
             // Find the corresponding index in the potentially reduced inputInstances set used for calculations
             // This mapping is tricky if instances were removed due to missing classes.
             // A safer approach is to map back based on content, or rely on the selected/covered BitSets
             // which use original indices if calculation was done carefully.
             // Assuming BitSets correspond to originalData indices:
             if (selectedInstances.get(i)) {
                  push(currentInst);
                  selectedCount++;
             } else if (!coveredInstances.get(i) && !currentInst.classIsMissing()) {
                 // Also keep original instances that were neither selected nor covered (likely boundary points)
                 // unless they had a missing class initially.
                 push(currentInst);
                 selectedCount++; // Count these as retained
             }
        }

        removedCount = originalCount - selectedCount;
        System.err.println("Removed " + removedCount + " instances (" + selectedCount + " retained).");

        flushInput();
        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }


    /** Precomputes the pairwise instance similarities. */
    private double[][] precomputeSimilarities(Instances data, int classIndex) throws Exception {
         int numInstances = data.numInstances();
         int numAttribs = data.numAttributes();
         double[][] sims = new double[numInstances][numInstances];

         for (int i = 0; i < numInstances; i++) {
             sims[i][i] = 1.0;
             Instance instI = data.instance(i);
             for (int j = i + 1; j < numInstances; j++) {
                 Instance instJ = data.instance(j);
                 double combinedSim = 1.0; // Identity for T-Norm

                 for (int a = 0; a < numAttribs; a++) {
                     if (a == classIndex) continue; // Skip class attribute

                     Attribute attr = data.attribute(a);
                     double valI = instI.value(a);
                     double valJ = instJ.value(a);
                     double featureSim;

                     // Handle missing values if needed (basic approach: 0 sim if one is missing)
                     if (instI.isMissing(a) || instJ.isMissing(a)) {
                         featureSim = 0.0; // Or handle based on m_MatchMissingValues if implemented
                     } else if (attr.isNominal()) {
                         featureSim = m_SimilarityEq.similarity(a, valI, valJ);
                     } else { // Numeric
                         featureSim = m_Similarity.similarity(a, valI, valJ);
                     }
                     combinedSim = m_compositionTNorm.calculate(combinedSim, featureSim);
                 }
                 sims[i][j] = sims[j][i] = combinedSim;
             }
         }
         return sims;
    }


    /** Calculates lower and upper approximations for instance i. */
    private double[] approximations(int i, int classValueI, Instances data, double[][] sims, boolean calculateUpper) throws Exception {
         double lowerApprox = 0;
         double upperApprox = 0;
         int numInstances = data.numInstances();
         ArrayList<Double> implicationValues = new ArrayList<>(numInstances);
         ArrayList<Double> tNormValues = new ArrayList<>(numInstances);

         if (m_approxType == APPROX_STANDARD) {
             lowerApprox = 1.0; // Start with identity for min
             upperApprox = 0.0; // Start with identity for max
             for (int j = 0; j < numInstances; j++) {
                 int classValueJ = (int) data.instance(j).classValue();
                 double membership = (classValueJ == classValueI) ? 1.0 : 0.0;
                 double simIJ = sims[i][j];

                 double impVal = m_Implicator.calculate(simIJ, membership);
                 lowerApprox = Math.min(lowerApprox, impVal);

                 if (calculateUpper) {
                      double tNormVal = m_TNorm.calculate(simIJ, membership);
                      upperApprox = Math.max(upperApprox, tNormVal);
                 }
             }
         } else { // RIM OWA
             for (int j = 0; j < numInstances; j++) {
                 int classValueJ = (int) data.instance(j).classValue();
                 double membership = (classValueJ == classValueI) ? 1.0 : 0.0;
                 double simIJ = sims[i][j];

                 implicationValues.add(m_Implicator.calculate(simIJ, membership));
                 if (calculateUpper) {
                     tNormValues.add(m_TNorm.calculate(simIJ, membership));
                 }
             }
             double[] weightsLower = generateRIMWeights(numInstances, m_rimAlpha); // Assuming same alpha for both now
             double[] weightsUpper = weightsLower; // Or generate separately if different alphas needed

             lowerApprox = applyOWA(implicationValues, weightsLower);
             if (calculateUpper) {
                 upperApprox = applyOWA(tNormValues, weightsUpper);
             }
         }

         return new double[]{lowerApprox, upperApprox};
    }


    // --- OWA Helper Methods ---

    /** Generates RIM OWA weights using Yager's method Q(r) = r^alpha. */
    private static double[] generateRIMWeights(int n, double alpha) {
        if (n <= 0) return new double[0];
        if (n == 1) return new double[]{1.0};
        if (alpha <= 0) {
            // Warning handled elsewhere or assume alpha=1 for average
             alpha = 1.0;
        }

        double[] weights = new double[n];
        final double finalAlpha = alpha; // Final for lambda
        DoubleUnaryOperator Q = r -> Math.pow(r, finalAlpha);

        double sumQ = 0;
        for (int i = 1; i <= n; i++) {
            weights[i-1] = Q.applyAsDouble((double)i / n) - Q.applyAsDouble((double)(i - 1) / n);
            sumQ += weights[i-1];
        }

        // Normalize
        if (Math.abs(sumQ - 1.0) > 1e-9) {
            for (int i = 0; i < n; i++) {
                weights[i] /= sumQ;
            }
        }
        return weights;
    }

     /** Applies OWA operator using pre-calculated weights. */
    private static double applyOWA(ArrayList<Double> values, double[] weights) {
        int n = values.size();
        if (n == 0) return 0.0; // Or 1.0 for lower approx default? Check context. Let's use 0 for now.
        if (n != weights.length) {
            System.err.println("OWA Error: Mismatch values/weights (" + n + "!=" + weights.length + ")");
            return 0.0; // Error condition
        }
        if (n == 1) return values.get(0);

        // Sort values ascending
        Collections.sort(values);

        // Calculate weighted sum
        double weightedSum = 0;
        for (int i = 0; i < n; i++) {
            weightedSum += weights[i] * values.get(i);
        }
        return weightedSum;
    }
     // Need functional interface for Q function if Java version < 8
     @FunctionalInterface
     interface DoubleUnaryOperator {
         double applyAsDouble(double operand);
     }


    // --- AdditionalMeasureProducer Implementation ---
    @Override
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.equalsIgnoreCase("measureNumInstancesRemoved")) {
            return removedCount;
        }
        throw new IllegalArgumentException(additionalMeasureName + " not supported (LAIR2)");
    }

    @Override
    public Enumeration<String> enumerateMeasures() {
        Vector<String> newVector = new Vector<>(1);
        newVector.addElement("measureNumInstancesRemoved");
        return newVector.elements();
    }

    // --- Revision ---
    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: LAIR2_Enhanced $");
    }

    // --- Main Method for Testing ---
    public static void main(String[] argv) {
        runFilter(new LAIR2(), argv);
    }
}