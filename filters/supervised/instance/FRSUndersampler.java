package weka.filters.supervised.instance;

import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.fuzzy.implicator.*; // Assuming these exist
import weka.fuzzy.similarity.*; // Assuming these exist
import weka.fuzzy.tnorm.*;       // Assuming these exist

import java.util.*;

/**
 <!-- globalinfo-start -->
 * Performs undersampling of the majority class using Fuzzy Rough Set (FRS) analysis.
 * Instances are evaluated based on their FRS Positive Region membership, which indicates
 * their dissimilarity to minority class instances. Different strategies are available
 * for removing majority instances based on this measure.
 * <p/>
 * Strategies: <br/>
 * - remove_safest: Removes majority instances with the highest POS membership (furthest from boundary). <br/>
 * - preserve_boundary: Keeps majority instances with POS membership below a threshold and removes randomly from the rest. <br/>
 * - weighted_removal: Removes majority instances probabilistically, with removal probability proportional to their POS membership. <br/>
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX(Concept): Based on using FRS instance evaluation for informed undersampling.
 * <pre>
 * @misc{frs_undersampling_concept,
 *    author = {Conceptual design based on FRS principles},
 *    title = {Fuzzy Rough Set Guided Undersampling},
 *    year = {2024}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -S <num>
 *  Specifies the random number seed (for methods involving randomness).
 *  (default 1)</pre>
 *
 * <pre> -STRATEGY <ratio|auto>
 *  Desired ratio of minority samples to majority samples after resampling.
 *  Specify as a double (e.g., 1.0 for 1:1). 'auto' defaults to 1.0.
 *  (default: auto)</pre>
 *
 * <pre> -UND_METHOD <remove_safest|preserve_boundary|weighted_removal>
 *  The FRS-guided undersampling strategy to use.
 *  (default: remove_safest)</pre>
 *
 * <pre> -BOUNDARY_THRESH <double>
 *  POS membership threshold used by 'preserve_boundary' method.
 *  Instances with POS <= threshold are kept as boundary points.
 *  (default: 0.3)</pre>
 *
 * <pre> -R <similarity specification>
 *  Similarity function for numeric features.
 *  (default: weka.fuzzy.similarity.Similarity3)</pre>
 *
 * <pre> -T <t-norm specification>
 *  T-norm for combining feature similarities.
 *  (default: weka.fuzzy.tnorm.TNormLukasiewicz)</pre>
 *
 * <pre> -POS_TYPE <standard|vqrs_avg|owa_linear>
 *  Method to calculate Positive Region Membership.
 *  (default: standard)</pre>
 *
 * <pre> -SIGMA <double>
 *  Sigma parameter for Gaussian similarity (if applicable).
 *  (default: 0.15)</pre>
 *
 <!-- options-end -->
 *
 * @author AI Assistant based on FRS concepts and Weka filter structure
 * @version 2025.04.24
 */
public class FRSUndersampler
        extends Filter
        implements SupervisedFilter, OptionHandler, TechnicalInformationHandler {

    /** For serialization. */
    static final long serialVersionUID = 8432987515135L;

    /** The random seed to use. */
    protected int m_RandomSeed = 1;
    /** Sampling strategy (ratio or 'auto'). */
    protected String m_SamplingStrategy = "auto";
    /** Undersampling method name. */
    protected String m_UndersamplingMethod = "remove_safest";
    /** Threshold for preserve_boundary method. */
    protected double m_BoundaryThreshold = 0.3;

    // --- FRS Components ---
    protected Similarity m_Similarity = new Similarity3();
    protected Similarity m_SimilarityEq = new SimilarityEq();
    protected TNorm m_TNorm = new TNormLukasiewicz();
    protected String m_PosRegionType = "standard";
    protected double m_Sigma = 0.15;

    /** Epsilon for numerical stability */
    protected static final double EPSILON = 1e-9;

    /** Stores the calculated positive region memberships */
    protected double[] m_PosMembership = null;
    /** Stores the precalculated similarity matrix */
    protected double[][] m_SimilarityMatrix = null;
    /** Min values for normalization */
    protected double[] m_MinArray = null;
    /** Max values for normalization */
    protected double[] m_MaxArray = null;


    /** Constructor */
    public FRSUndersampler() {}

    /**
     * Returns a string describing this filter.
     * @return a description of the filter suitable for displaying in the GUI.
     */
    public String globalInfo() {
        return "Performs undersampling of the majority class using Fuzzy Rough Set (FRS) analysis.\n"
                + "Evaluates majority instances based on FRS Positive Region membership (dissimilarity to minority classes) "
                + "and removes instances according to selected strategy (remove_safest, preserve_boundary, weighted_removal).";
    }

    /**
     * Returns an instance of a TechnicalInformation object.
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.MISC);
        result.setValue(Field.AUTHOR, "Conceptual design based on FRS principles");
        result.setValue(Field.TITLE, "Fuzzy Rough Set Guided Undersampling");
        result.setValue(Field.YEAR, "2024");
        result.setValue(Field.NOTE, "Implements FRS-based evaluation for targeted majority class removal.");
        return result;
    }

    /**
     * Returns the revision string.
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: FRSUndersampler_V1 $");
    }

    /**
     * Returns the Capabilities of this filter.
     * @return the capabilities of this object
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // Attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
        // Class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);
        // Instances
        result.setMinimumNumberInstances(2); // Need at least one of each class ideally
        return result;
    }

    /**
     * Returns an enumeration describing the available options.
     * @return an enumeration of all the available options.
     */
    public Enumeration<Option> listOptions() {
        Vector<Option> options = new Vector<>();
        options.addElement(new Option("\tRandom number seed (for random steps).\n\t(default 1)", "S", 1, "-S <num>"));
        options.addElement(new Option("\tDesired ratio Min/Maj (e.g., 1.0) or 'auto'.\n\t(default: auto = 1.0)", "STRATEGY", 1, "-STRATEGY <ratio|auto>"));
        options.addElement(new Option("\tUndersampling strategy.\n\t(remove_safest|preserve_boundary|weighted_removal, default: remove_safest)", "UND_METHOD", 1, "-UND_METHOD <string>"));
        options.addElement(new Option("\tPOS threshold for preserve_boundary.\n\t(default: 0.3)", "BOUNDARY_THRESH", 1, "-BOUNDARY_THRESH <double>"));
        options.addElement(new Option("\tSimilarity function for numeric features.\n\t(default: weka.fuzzy.similarity.Similarity3)", "R", 1, "-R <similarity specification>"));
        options.addElement(new Option("\tT-norm for combining feature similarities.\n\t(default: weka.fuzzy.tnorm.TNormLukasiewicz)", "T", 1, "-T <t-norm specification>"));
        options.addElement(new Option("\tMethod for calculating Positive Region Membership.\n\t(standard|vqrs_avg|owa_linear, default: standard)", "POS_TYPE", 1, "-POS_TYPE <string>"));
        options.addElement(new Option("\tSigma for Gaussian similarity (if applicable).\n\t(default: 0.15)", "SIGMA", 1, "-SIGMA <double>"));
        return options.elements();
    }

    /**
     * Parses a given list of options.
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        String seedStr = Utils.getOption("S", options);
        setRandomSeed(seedStr.length() != 0 ? Integer.parseInt(seedStr) : 1);

        String strategyStr = Utils.getOption("STRATEGY", options);
        setSamplingStrategy(strategyStr.length() != 0 ? strategyStr : "auto");

        String methodStr = Utils.getOption("UND_METHOD", options);
        setUndersamplingMethod(methodStr.length() != 0 ? methodStr : "remove_safest");

        String threshStr = Utils.getOption("BOUNDARY_THRESH", options);
        setBoundaryThreshold(threshStr.length() != 0 ? Double.parseDouble(threshStr) : 0.3);

        String simString = Utils.getOption('R', options);
        setSimilarity((Similarity) Utils.forName(Similarity.class, simString.length() != 0 ? simString : Similarity3.class.getName(), null));

        String tnormString = Utils.getOption('T', options);
        setTNorm((TNorm) Utils.forName(TNorm.class, tnormString.length() != 0 ? tnormString : TNormLukasiewicz.class.getName(), null));

        String posTypeStr = Utils.getOption("POS_TYPE", options);
        setPosRegionType(posTypeStr.length() != 0 ? posTypeStr : "standard");

        String sigmaStr = Utils.getOption("SIGMA", options);
        setSigma(sigmaStr.length() != 0 ? Double.parseDouble(sigmaStr) : 0.15);

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of the filter.
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        Vector<String> options = new Vector<>();
        options.add("-S"); options.add("" + getRandomSeed());
        options.add("-STRATEGY"); options.add(getSamplingStrategy());
        options.add("-UND_METHOD"); options.add(getUndersamplingMethod());
        options.add("-BOUNDARY_THRESH"); options.add("" + getBoundaryThreshold());
        options.add("-R"); options.add(getSimilarity().getClass().getName());
        options.add("-T"); options.add(getTNorm().getClass().getName());
        options.add("-POS_TYPE"); options.add(getPosRegionType());
        options.add("-SIGMA"); options.add("" + getSigma());
        return options.toArray(new String[0]);
    }

    // --- Getters and Setters for Options (with TipText) ---

    public String randomSeedTipText() { return "The seed used for random sampling steps."; }
    public int getRandomSeed() { return m_RandomSeed; }
    public void setRandomSeed(int value) { m_RandomSeed = value; }

    public String samplingStrategyTipText() { return "Desired ratio Minority/Majority (e.g., 1.0) or 'auto' (1.0)."; }
    public String getSamplingStrategy() { return m_SamplingStrategy; }
    public void setSamplingStrategy(String value) { m_SamplingStrategy = value; }

    public String undersamplingMethodTipText() { return "Undersampling strategy (remove_safest | preserve_boundary | weighted_removal)."; }
    public String getUndersamplingMethod() { return m_UndersamplingMethod; }
    public void setUndersamplingMethod(String value) {
        if (value.equalsIgnoreCase("remove_safest") || value.equalsIgnoreCase("preserve_boundary") || value.equalsIgnoreCase("weighted_removal")) {
             m_UndersamplingMethod = value.toLowerCase();
        } else {
             throw new IllegalArgumentException("Invalid UND_METHOD: " + value);
        }
    }

    public String boundaryThresholdTipText() { return "POS membership threshold for 'preserve_boundary' method."; }
    public double getBoundaryThreshold() { return m_BoundaryThreshold; }
    public void setBoundaryThreshold(double value) { m_BoundaryThreshold = value; }

    public String similarityTipText() { return "Similarity function for numeric features."; }
    public Similarity getSimilarity() { return m_Similarity; }
    public void setSimilarity(Similarity s) { m_Similarity = s; }

    public String TNormTipText() { return "T-norm for combining feature similarities."; }
    public TNorm getTNorm() { return m_TNorm; }
    public void setTNorm(TNorm t) { m_TNorm = t; }

    public String posRegionTypeTipText() { return "Method for calculating Positive Region Membership (standard | vqrs_avg | owa_linear)."; }
    public String getPosRegionType() { return m_PosRegionType; }
    public void setPosRegionType(String type) {
        if (type.equalsIgnoreCase("standard") || type.equalsIgnoreCase("vqrs_avg") || type.equalsIgnoreCase("owa_linear")) {
            m_PosRegionType = type.toLowerCase();
        } else {
            throw new IllegalArgumentException("Invalid POS_TYPE: " + type);
        }
    }

    public String sigmaTipText() { return "Sigma parameter for Gaussian similarity (if applicable)."; }
    public double getSigma() { return m_Sigma; }
    public void setSigma(double s) { m_Sigma = s; }


    // --- Filter Implementation ---

    /**
     * Sets the format of the input instances.
     * @param instanceInfo an Instances object containing the input instance structure.
     * @return true as output format is set immediately.
     * @throws Exception if the input format can't be set successfully.
     */
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        super.setInputFormat(instanceInfo);
        // Output format has same structure, just fewer instances
        setOutputFormat(instanceInfo);
         m_PosMembership = null; // Reset internal state
         m_SimilarityMatrix = null;
         m_MinArray = null;
         m_MaxArray = null;
        return true;
    }

    /**
     * Input an instance for filtering. Requires all instances be read before producing output.
     * @param instance the input instance
     * @return false, as output depends on batch finishing.
     * @throws IllegalStateException if no input structure has been defined.
     */
    public boolean input(Instance instance) {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }
        bufferInput(instance);
        return false;
    }

    /**
     * Signifies that this batch of input to the filter is finished.
     * @return true if there are instances pending output.
     * @throws Exception if filtering fails.
     */
    public boolean batchFinished() throws Exception {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        if (!m_FirstBatchDone) {
             Instances inputData = getInputFormat();
             Instances workingData = new Instances(inputData);
             workingData.deleteWithMissingClass();

             if (workingData.numInstances() < 2 || workingData.numClasses() <= 1) {
                  System.err.println("Warning: Not enough instances or classes after removing missing class values to perform FRS Undersampling. Passing original data through.");
                  for (int i = 0; i < inputData.numInstances(); i++) {
                      push(inputData.instance(i));
                  }
             } else {
                 doFRSUndersampling(workingData);
             }
        }

        flushInput();
        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }

    /**
     * The procedure implementing the FRS Undersampling algorithm.
     * Pushes selected instances to the output queue.
     * @param data The input instances (must have class attribute, no missing classes).
     * @throws Exception if processing fails.
     */
    protected void doFRSUndersampling(Instances data) throws Exception {
         int numInstances = data.numInstances();
         int classIndex = data.classIndex();

         // --- 1. Identify Majority/Minority Classes and Counts ---
         int[] counts = data.attributeStats(classIndex).nominalCounts;
         int majorityClassIndex = -1;
         int majoritySize = -1;
         int totalMinoritySize = 0;

         for (int i = 0; i < counts.length; i++) {
              if (counts[i] > majoritySize) {
                   majoritySize = counts[i];
                   majorityClassIndex = i;
              }
         }
         // Calculate total minority size
         for (int i = 0; i < counts.length; i++) {
             if (i != majorityClassIndex) {
                 totalMinoritySize += counts[i];
             }
         }

         if (majorityClassIndex == -1 || totalMinoritySize == 0) {
              System.err.println("Warning: Could not identify distinct majority/minority classes. Passing data through.");
              for (int i = 0; i < data.numInstances(); i++) push(data.instance(i));
              return;
         }
          System.err.println("FRSUndersampler: Majority class index: " + majorityClassIndex + " (Size: " + majoritySize + "), Total minority size: " + totalMinoritySize);


         // --- 2. Determine Target Majority Count ---
         double targetRatio = 1.0; // Default for auto
         if (!m_SamplingStrategy.equalsIgnoreCase("auto")) {
             try {
                 targetRatio = Double.parseDouble(m_SamplingStrategy);
                 if (targetRatio <= 0) {
                      System.err.println("Warning: Target ratio must be positive. Using 1.0.");
                      targetRatio = 1.0;
                 }
             } catch (NumberFormatException e) {
                 System.err.println("Warning: Invalid sampling strategy '" + m_SamplingStrategy + "'. Using 'auto' (1.0).");
                 targetRatio = 1.0;
             }
         }
         // Target number = Total Minority / Ratio
         int targetMajorityCount = Math.max(0, (int) Math.round((double) totalMinoritySize / targetRatio));
         int numToRemove = Math.max(0, majoritySize - targetMajorityCount);

         // Ensure we don't remove all unless target is 0
         if (targetMajorityCount == 0) {
             numToRemove = majoritySize;
         } else {
              numToRemove = Math.min(numToRemove, majoritySize - 1); // Keep at least 1 if target > 0
         }
         targetMajorityCount = majoritySize - numToRemove; // Recalculate actual target count

         System.err.println("FRSUndersampler: Target ratio: " + String.format("%.2f", targetRatio) + ", Target majority count: " + targetMajorityCount + ", Removing: " + numToRemove);

         if (numToRemove == 0) {
             System.err.println("FRSUndersampler: No instances to remove.");
             for (int i = 0; i < data.numInstances(); i++) push(data.instance(i));
             return;
         }

         // --- 3. Prepare FRS Components & Data ---
         setupFuzzyComponents(data);
         Instances normalizedData = normalizeData(data);
         System.err.println("FRSUndersampler: Calculating similarity matrix...");
         m_SimilarityMatrix = calculateSimilarityMatrix(normalizedData);
         System.err.println("FRSUndersampler: Calculating positive region membership...");
         m_PosMembership = calculatePositiveRegionMembership(normalizedData, m_SimilarityMatrix);

         // --- 4. Identify Majority Instances and Apply Strategy ---
         ArrayList<Pair> majorityPairs = new ArrayList<>(); // Pair(index, pos_membership)
         ArrayList<Integer> minorityIndices = new ArrayList<>();
         for (int i = 0; i < numInstances; i++) {
             if ((int) data.instance(i).classValue() == majorityClassIndex) {
                 majorityPairs.add(new Pair(i, m_PosMembership[i]));
             } else {
                 minorityIndices.add(i);
             }
         }

         Set<Integer> keptMajorityIndicesSet = new HashSet<>();
         Random random = new Random(m_RandomSeed);

         if (m_UndersamplingMethod.equals("remove_safest")) {
             // Sort descending by POS score
             Collections.sort(majorityPairs, Collections.reverseOrder());
             // Keep the ones with lower scores (at the end after sorting)
             for (int i = numToRemove; i < majorityPairs.size(); i++) {
                 keptMajorityIndicesSet.add(majorityPairs.get(i).index);
             }
         }
         else if (m_UndersamplingMethod.equals("preserve_boundary")) {
             ArrayList<Integer> boundaryIndices = new ArrayList<>();
             ArrayList<Integer> coreIndices = new ArrayList<>();
             for (Pair p : majorityPairs) {
                 if (p.value <= m_BoundaryThreshold) {
                     boundaryIndices.add(p.index);
                 } else {
                     coreIndices.add(p.index);
                 }
             }
             // Keep all boundary points
             keptMajorityIndicesSet.addAll(boundaryIndices);
             int numCoreToKeep = Math.max(0, targetMajorityCount - boundaryIndices.size());

             if (numCoreToKeep >= coreIndices.size()) {
                 keptMajorityIndicesSet.addAll(coreIndices); // Keep all core
             } else {
                 // Randomly select core indices to keep
                 Collections.shuffle(coreIndices, random);
                 for (int i = 0; i < numCoreToKeep; i++) {
                     keptMajorityIndicesSet.add(coreIndices.get(i));
                 }
             }
         }
         else if (m_UndersamplingMethod.equals("weighted_removal")) {
              // Select indices TO REMOVE, weighted by POS score
              List<Pair> removalCandidates = new ArrayList<>(majorityPairs);
              Set<Integer> removedIndices = new HashSet<>();
              for(int i=0; i<numToRemove; i++) {
                  Pair removedPair = weightedRandomChoice(removalCandidates, random);
                  if (removedPair != null) {
                       removedIndices.add(removedPair.index);
                       // Remove from candidates for next iteration (inefficient but simple)
                       removalCandidates.removeIf(p -> p.index == removedPair.index);
                  } else {
                       break; // No more candidates to remove
                  }
              }
              // Keep indices that were NOT removed
              for(Pair p : majorityPairs) {
                  if (!removedIndices.contains(p.index)) {
                       keptMajorityIndicesSet.add(p.index);
                  }
              }
         }
         else { // Should not happen
              System.err.println("Warning: Unknown undersampling method. Keeping all majority.");
              for(Pair p : majorityPairs) keptMajorityIndicesSet.add(p.index);
         }

         System.err.println("FRSUndersampler: Kept " + keptMajorityIndicesSet.size() + " majority instances.");

         // --- 5. Output Selected Instances ---
         // Output all minority instances first
         for (int idx : minorityIndices) {
             push(data.instance(idx));
         }
         // Output the kept majority instances
         for (int idx : keptMajorityIndicesSet) {
             push(data.instance(idx));
         }
    }


    // --- Helper methods (setupFuzzyComponents, normalizeData, calculateSimilarityMatrix, calculatePositiveRegionMembership, weightedRandomChoice, Pair, OWAUtils) ---
    // --- These are identical to the ones in FRSMOTE.java - Copy them here ---

     /** Helper to set up fuzzy components based on data */
     protected void setupFuzzyComponents(Instances data) {
         m_Similarity.setInstances(data);
         m_SimilarityEq.setInstances(data);

     }

     /** Helper for Min-Max normalization */
     protected Instances normalizeData(Instances data) {
         m_MinArray = new double[data.numAttributes()];
         m_MaxArray = new double[data.numAttributes()];
         Instances normData = new Instances(data); // Copy structure

         for (int j = 0; j < data.numAttributes(); j++) {
             if (data.attribute(j).isNumeric() && j != data.classIndex()) {
                 m_MinArray[j] = data.attributeStats(j).numericStats.min;
                 m_MaxArray[j] = data.attributeStats(j).numericStats.max;
             }
         }

         for (int i = 0; i < data.numInstances(); i++) {
             double[] normValues = new double[data.numAttributes()];
             Instance inst = data.instance(i);
             for (int j = 0; j < data.numAttributes(); j++) {
                 if (inst.isMissing(j)) {
                     normValues[j] = Utils.missingValue();
                 } else if (data.attribute(j).isNumeric() && j != data.classIndex()) {
                     double range = m_MaxArray[j] - m_MinArray[j];
                     if (range < EPSILON) {
                         normValues[j] = 0.5; // Handle constant attribute
                     } else {
                         normValues[j] = (inst.value(j) - m_MinArray[j]) / range;
                     }
                 } else {
                     normValues[j] = inst.value(j); // Keep nominal/class/date as is
                 }
             }
             Instance normInstance = new DenseInstance(inst.weight(), normValues);
             normInstance.setDataset(data); // Keep dataset reference
             normData.add(normInstance);
         }
         return normData;
     }

     /** Helper to calculate the similarity matrix (Optimized Logic) */
     protected double[][] calculateSimilarityMatrix(Instances normData) throws Exception {
         int n = normData.numInstances();
         int m = normData.numAttributes();
         int classIdx = normData.classIndex();
         double[][] simMatrix = new double[n][n];

         for (int i = 0; i < n; i++) {
             simMatrix[i][i] = 1.0;
             Instance instI = normData.instance(i);
             for (int j = i + 1; j < n; j++) {
                 Instance instJ = normData.instance(j);
                 double combinedSim = 1.0; // Identity for T-Norm

                 for (int a = 0; a < m; a++) {
                     if (a == classIdx) continue;
                     Attribute attr = normData.attribute(a);
                     double valI = instI.value(a);
                     double valJ = instJ.value(a);
                     double featureSim = 0.0;

                     if (instI.isMissing(a) || instJ.isMissing(a)) {
                         featureSim = 0.0; // Simplistic handling
                     } else if (attr.isNumeric() || attr.isDate()) { // Treat Date as Numeric for Similarity
                         featureSim = m_Similarity.similarity(a, valI, valJ);
                     } else if (attr.isNominal()) {
                         featureSim = m_SimilarityEq.similarity(a, valI, valJ); // Use equals for nominal
                     }
                     combinedSim = m_TNorm.calculate(combinedSim, featureSim);
                 }
                 simMatrix[i][j] = simMatrix[j][i] = combinedSim;
             }
         }
         return simMatrix;
     }

     /** Helper to calculate positive region membership (Optimized Logic) */
     protected double[] calculatePositiveRegionMembership(Instances normData, double[][] simMatrix) {
         int n = normData.numInstances();
         int classIdx = normData.classIndex();
         double[] posMembership = new double[n];

         for (int i = 0; i < n; i++) {
             double p1Class = normData.instance(i).classValue();
             ArrayList<Double> enemySimilarities = new ArrayList<>();

             for (int j = 0; j < n; j++) {
                 if (i == j) continue;
                 double p2Class = normData.instance(j).classValue();
                 if (p1Class != p2Class) {
                     enemySimilarities.add(simMatrix[i][j]);
                 }
             }

             if (enemySimilarities.isEmpty()) {
                 posMembership[i] = 1.0;
             } else {
                 if (m_PosRegionType.equals("standard")) {
                     double maxEnemySim = Collections.max(enemySimilarities);
                     posMembership[i] = 1.0 - maxEnemySim;
                 } else if (m_PosRegionType.equals("vqrs_avg")) {
                     double sumEnemySim = 0;
                     for(double sim : enemySimilarities) sumEnemySim += sim;
                     posMembership[i] = 1.0 - (sumEnemySim / enemySimilarities.size());
                 } else if (m_PosRegionType.equals("owa_linear")) {
                     ArrayList<Double> dissimilarities = new ArrayList<>();
                     for(double sim : enemySimilarities) dissimilarities.add(1.0 - sim);
                     Collections.sort(dissimilarities); // Ascending sort
                     double[] weights = FRSMOTE.OWAUtils.owaWeightsLinear(dissimilarities.size()); // Use shared util
                     double owaSum = 0;
                     for (int k = 0; k < dissimilarities.size(); k++) {
                         owaSum += weights[k] * dissimilarities.get(k);
                     }
                     posMembership[i] = owaSum;
                 } else { // Default to standard
                     double maxEnemySim = Collections.max(enemySimilarities);
                     posMembership[i] = 1.0 - maxEnemySim;
                 }
             }
             // Clamp and check NaN
             posMembership[i] = Math.max(0.0, Math.min(1.0, posMembership[i]));
             if (Double.isNaN(posMembership[i])) posMembership[i] = 0.0;
         }
         return posMembership;
     }

     /** Helper for weighted random choice */
     protected Pair weightedRandomChoice(List<Pair> itemsWithWeights, Random random) {
        double totalWeight = 0;
        // Filter out non-positive weights for selection pool
        List<Pair> validCandidates = new ArrayList<>();
        for (Pair p : itemsWithWeights) {
             if (p.value > EPSILON) {
                 validCandidates.add(p);
                 totalWeight += p.value;
             }
        }
        if (validCandidates.isEmpty()) {
             // Fallback: Uniform random choice from original list if no positive weights
             if (itemsWithWeights.isEmpty()) return null;
             return itemsWithWeights.get(random.nextInt(itemsWithWeights.size()));
        }

        double randomVal = random.nextDouble() * totalWeight;
        double cumulativeWeight = 0;
        for (Pair p : validCandidates) {
            cumulativeWeight += p.value;
            if (cumulativeWeight >= randomVal) {
                return p;
            }
        }
        // Fallback for floating point issues
        return validCandidates.get(validCandidates.size() - 1);
     }

     /** Simple Pair class to hold index and value */
     protected static class Pair implements Comparable<Pair> {
         int index;
         double value;
         Pair(int index, double value) { this.index = index; this.value = value; }
         // Compare based on value (descending)
         @Override public int compareTo(Pair other) { return Double.compare(other.value, this.value); } // Note: Reverse for descending sort
     }

    /**
     * Main method for running this filter.
     * @param args should contain arguments to the filter: use -h for help
     */
    public static void main(String[] args) {
        runFilter(new FRSUndersampler(), args);
    }
}