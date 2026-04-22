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
 * Resamples a dataset by applying the Fuzzy Rough Set based Synthetic Minority Oversampling TEchnique (FRSMOTE).
 * This version uses FRS Positive Region membership to guide the selection of minority instances for SMOTE.
 * It prioritizes instances considered more "certain" or "safe" based on their dissimilarity to enemy classes.
 * Supports different fuzzy similarity functions, T-Norms, and POS region calculation methods.
 * Optionally biases the interpolation towards the parent point with higher certainty.
 * <p/>
 * Based on concepts from: <br/>
 * Nitesh V. Chawla et. al. (2002). SMOTE. JAIR.<br/>
 * And incorporating Fuzzy Rough Set principles for instance weighting.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX(SMOTE):
 * <pre>
 * @article{al.2002,
 *    author = {Nitesh V. Chawla et. al.},
 *    journal = {Journal of Artificial Intelligence Research},
 *    pages = {321-357},
 *    title = {Synthetic Minority Over-sampling Technique},
 *    volume = {16},
 *    year = {2002}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -S <num>
 *  Specifies the random number seed.
 *  (default 1)</pre>
 *
 * <pre> -P <percentage>
 *  Specifies percentage of SMOTE instances to create relative to the original number of minority instances.
 *  (default 100.0)</pre>
 *
 * <pre> -K <nearest-neighbors>
 *  Specifies the number of nearest neighbors (based on fuzzy similarity) to use for selecting the SMOTE partner.
 *  (default 5)</pre>
 *
 * <pre> -C <value-index>
 *  Specifies the index (1-based) of the nominal class value to SMOTE.
 *  Use '0' or 'auto' to auto-detect the non-empty minority class.
 *  (default 0)</pre>
 *
 * <pre> -R <similarity specification>
 *  Similarity function to use for numeric features.
 *  (default: weka.fuzzy.similarity.Similarity3)</pre>
 *
 * <pre> -T <t-norm specification>
 *  T-norm to use for combining feature similarities.
 *  (default: weka.fuzzy.tnorm.TNormLukasiewicz)</pre>
 *
 * <pre> -POS_TYPE <standard|vqrs_avg|owa_linear>
 *  Method to calculate Positive Region Membership.
 *  (default: standard)</pre>
 *
 * <pre> -SIGMA <double>
 *  Sigma parameter for Gaussian similarity (if -R specifies it).
 *  (default: 0.15)</pre>
 *
 * <pre> -BIAS_INTERPOLATION <true|false>
 *  Bias SMOTE interpolation towards parent with higher POS membership.
 *  (default: false)</pre>
 *
 <!-- options-end -->
 *
 * @author Based on original SMOTE by Ryan Lichtenwalter, adapted for FRS by AI Assistant
 * @version 2025.04.24
 */
public class FRSMOTE
        extends Filter
        implements SupervisedFilter, OptionHandler, TechnicalInformationHandler {

    /** For serialization. */
    static final long serialVersionUID = -7854875158749845L;

    /** The number of nearest neighbors (based on fuzzy similarity) to use. */
    protected int m_NearestNeighbors = 5;
    /** The random seed to use. */
    protected int m_RandomSeed = 1;
    /** The percentage of SMOTE instances to create. */
    protected double m_Percentage = 100.0;
    /** The index of the class value (1-based string). */
    protected String m_ClassValueIndex = "0"; // 0 means auto-detect
    /** Whether to detect the minority class automatically. */
    protected boolean m_DetectMinorityClass = true;

    // --- FRS Components ---
    /** Similarity function for numeric features */
    protected Similarity m_Similarity = new Similarity3();
    /** Similarity function for nominal features (usually equals) */
    protected Similarity m_SimilarityEq = new SimilarityEq();
    /** T-norm for combining feature similarities */
    protected TNorm m_TNorm = new TNormLukasiewicz(); // Used for combining sims
    /** Method for calculating POS membership */
    protected String m_PosRegionType = "standard"; // standard, vqrs_avg, owa_linear
    /** Sigma for Gaussian similarity */
    protected double m_Sigma = 0.15; // Default, only used if Similarity needs it
    /** Bias interpolation based on POS membership */
    protected boolean m_BiasInterpolation = false;

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
    public FRSMOTE() { }

    /**
     * Returns a string describing this filter.
     * @return a description of the filter suitable for displaying in the GUI.
     */
    public String globalInfo() {
        return "Resamples a dataset by applying the Fuzzy Rough Set based Synthetic Minority Oversampling Technique (FRSMOTE).\n"
                + "Uses FRS Positive Region membership to weight the selection of minority instances.\n"
                + "Handles numeric and nominal attributes using fuzzy similarity.\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed information about the technical background of this class.
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Nitesh V. Chawla et. al. (SMOTE basis), FRS concepts integrated");
        result.setValue(Field.TITLE, "SMOTE: Synthetic Minority Over-sampling Technique (with FRS modifications)");
        result.setValue(Field.JOURNAL, "Journal of Artificial Intelligence Research (for original SMOTE)");
        result.setValue(Field.YEAR, "2002 (for original SMOTE)");
        result.setValue(Field.VOLUME, "16");
        result.setValue(Field.PAGES, "321-357");
        result.setValue(Field.NOTE, "FRS modifications incorporate instance weighting based on fuzzy positive region membership.");
        return result;
    }

    /**
     * Returns the revision string.
     * @return the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: FRSMOTE_V1 $");
    }

    /**
     * Returns the Capabilities of this filter.
     * @return            the capabilities of this object
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // Attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES); // SMOTE handles Date like numeric
        result.enable(Capability.MISSING_VALUES); // Handled by similarity functions / interpolation
        // Class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES); // Filtered out before processing
        // Instances
        result.setMinimumNumberInstances(2); // Need at least 2 for SMOTE
        return result;
    }

    /**
     * Returns an enumeration describing the available options.
     * @return an enumeration of all the available options.
     */
    public Enumeration<Option> listOptions() {
        Vector<Option> options = new Vector<>();
        options.addElement(new Option("\tSpecifies the random number seed.\n\t(default 1)", "S", 1, "-S <num>"));
        options.addElement(new Option("\tSpecifies percentage of SMOTE instances to create.\n\t(default 100.0)", "P", 1, "-P <percentage>"));
        options.addElement(new Option("\tSpecifies the number of nearest neighbors to use.\n\t(based on fuzzy similarity, default 5)", "K", 1, "-K <nearest-neighbors>"));
        options.addElement(new Option("\tSpecifies the index (1-based) of the nominal class value to SMOTE.\n\t(default 0 = auto-detect non-empty minority class)", "C", 1, "-C <value-index>"));
        options.addElement(new Option("\tSimilarity function for numeric features.\n\t(default: weka.fuzzy.similarity.Similarity3)", "R", 1, "-R <similarity specification>"));
        options.addElement(new Option("\tT-norm for combining feature similarities.\n\t(default: weka.fuzzy.tnorm.TNormLukasiewicz)", "T", 1, "-T <t-norm specification>"));
        options.addElement(new Option("\tMethod for calculating Positive Region Membership.\n\t(standard | vqrs_avg | owa_linear, default: standard)", "POS_TYPE", 1, "-POS_TYPE <string>"));
        options.addElement(new Option("\tSigma for Gaussian similarity (if applicable).\n\t(default: 0.15)", "SIGMA", 1, "-SIGMA <double>"));
        options.addElement(new Option("\tBias SMOTE interpolation towards parent with higher POS membership.\n\t(default: false)", "BIAS_INTERPOLATION", 1, "-BIAS_INTERPOLATION <true|false>"));
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

        String percentageStr = Utils.getOption("P", options);
        setPercentage(percentageStr.length() != 0 ? Double.parseDouble(percentageStr) : 100.0);

        String nnStr = Utils.getOption("K", options);
        setNearestNeighbors(nnStr.length() != 0 ? Integer.parseInt(nnStr) : 5);

        String classValueIndexStr = Utils.getOption("C", options);
        setClassValue(classValueIndexStr.length() != 0 ? classValueIndexStr : "0");

        String simString = Utils.getOption('R', options);
        setSimilarity((Similarity) Utils.forName(Similarity.class, simString.length() != 0 ? simString : Similarity3.class.getName(), null));

        String tnormString = Utils.getOption('T', options);
        setTNorm((TNorm) Utils.forName(TNorm.class, tnormString.length() != 0 ? tnormString : TNormLukasiewicz.class.getName(), null));

        String posTypeStr = Utils.getOption("POS_TYPE", options);
        setPosRegionType(posTypeStr.length() != 0 ? posTypeStr : "standard");

        String sigmaStr = Utils.getOption("SIGMA", options);
        setSigma(sigmaStr.length() != 0 ? Double.parseDouble(sigmaStr) : 0.15);

        String biasStr = Utils.getOption("BIAS_INTERPOLATION", options);
        setBiasInterpolation(biasStr.equalsIgnoreCase("true"));

        Utils.checkForRemainingOptions(options);
    }

    /**
     * Gets the current settings of the filter.
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        Vector<String> options = new Vector<>();
        options.add("-S"); options.add("" + getRandomSeed());
        options.add("-P"); options.add("" + getPercentage());
        options.add("-K"); options.add("" + getNearestNeighbors());
        options.add("-C"); options.add(getClassValue());
        options.add("-R"); options.add(getSimilarity().getClass().getName());
        options.add("-T"); options.add(getTNorm().getClass().getName());
        options.add("-POS_TYPE"); options.add(getPosRegionType());
        options.add("-SIGMA"); options.add("" + getSigma());
        options.add("-BIAS_INTERPOLATION"); options.add("" + getBiasInterpolation());
        return options.toArray(new String[0]);
    }

    // --- Getters and Setters for Options (with TipText) ---

    public String randomSeedTipText() { return "The seed used for random sampling."; }
    public int getRandomSeed() { return m_RandomSeed; }
    public void setRandomSeed(int value) { m_RandomSeed = value; }

    public String percentageTipText() { return "The percentage of SMOTE instances to create relative to the original minority class size."; }
    public double getPercentage() { return m_Percentage; }
    public void setPercentage(double value) {
        if (value >= 0) m_Percentage = value;
        else throw new IllegalArgumentException("Percentage must be >= 0!");
    }

    public String nearestNeighborsTipText() { return "The number of nearest neighbors (based on fuzzy similarity) to use."; }
    public int getNearestNeighbors() { return m_NearestNeighbors; }
    public void setNearestNeighbors(int value) {
        if (value >= 1) m_NearestNeighbors = value;
        else throw new IllegalArgumentException("At least 1 neighbor necessary!");
    }

    public String classValueTipText() { return "The index (1-based) of the class value to which SMOTE should be applied. Use '0' or 'auto' to auto-detect the non-empty minority class."; }
    public String getClassValue() { return m_ClassValueIndex; }
    public void setClassValue(String value) {
        m_ClassValueIndex = value;
        m_DetectMinorityClass = value.equals("0") || value.equalsIgnoreCase("auto");
    }

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

    public String biasInterpolationTipText() { return "Bias SMOTE interpolation towards parent with higher POS membership."; }
    public boolean getBiasInterpolation() { return m_BiasInterpolation; }
    public void setBiasInterpolation(boolean b) { m_BiasInterpolation = b; }

    // --- Filter Implementation ---

    /**
     * Sets the format of the input instances.
     * @param instanceInfo an Instances object containing the input instance structure.
     * @return true if the outputFormat may be collected immediately.
     * @throws Exception if the input format can't be set successfully.
     */
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        super.setInputFormat(instanceInfo);
        super.setOutputFormat(instanceInfo); // Output format is same structure
        m_PosMembership = null; // Reset internal state
        m_SimilarityMatrix = null;
        m_MinArray = null;
        m_MaxArray = null;
        return true;
    }

    /**
     * Input an instance for filtering. Filter requires all training instances
     * be read before producing output.
     * @param instance the input instance
     * @return true if the filtered instance may now be collected with output().
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
        // Buffer all instances
        bufferInput(instance);
        return false;
    }

    /**
     * Signifies that this batch of input to the filter is finished.
     * @return true if there are instances pending output.
     * @throws Exception if provided options cannot be executed on input instances.
     */
    public boolean batchFinished() throws Exception {
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        if (!m_FirstBatchDone) {
            // Perform FRSMOTE
            Instances inputData = getInputFormat();
            Instances workingData = new Instances(inputData); // Copy to work on
            workingData.deleteWithMissingClass(); // SMOTE needs class labels

            if (workingData.numInstances() < 2) {
                 System.err.println("Warning: Not enough instances (" + workingData.numInstances() + ") after removing missing class values to perform FRSMOTE. Passing original data through.");
            } else if (workingData.numClasses() <= 1) {
                 System.err.println("Warning: Data has only one class label. Cannot perform FRSMOTE. Passing original data through.");
            } else {
                doFRSMOTE(workingData); // Perform SMOTE on data without missing class
                // Original instances are pushed inside doFRSMOTE
            }

             // If we didn't run doFRSMOTE or it failed early, push original data
             if (numPendingOutput() == 0) {
                 for (int i = 0; i < inputData.numInstances(); i++) {
                     push(inputData.instance(i));
                 }
             }
        }

        flushInput(); // Clear the buffer
        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }

    /**
     * The procedure implementing the FRSMOTE algorithm.
     * @param data The input instances (must have class attribute, no missing classes).
     * @throws Exception if processing fails.
     */
    protected void doFRSMOTE(Instances data) throws Exception {
        int numInstances = data.numInstances();
        int classIndex = data.classIndex();
        Attribute classAttr = data.attribute(classIndex);

        // --- 1. Identify Minority Class ---
        int minorityClassIndex = -1;
        int minSize = Integer.MAX_VALUE;

        if (m_DetectMinorityClass) {
            int[] counts = data.attributeStats(classIndex).nominalCounts;
            for (int i = 0; i < counts.length; i++) {
                if (counts[i] > 0 && counts[i] < minSize) {
                    minSize = counts[i];
                    minorityClassIndex = i;
                }
            }
            if (minorityClassIndex == -1) { // Handle case where all classes have same size or only one class present
                 System.err.println("Warning: Could not automatically detect a unique minority class. Using class index 0.");
                 minorityClassIndex = 0;
                 minSize = counts[0]; // Need minSize for K check
            }
        } else {
            try {
                int idx = Integer.parseInt(m_ClassValueIndex);
                if (idx <= 0 || idx > data.numClasses()) {
                    throw new Exception("Class value index '" + m_ClassValueIndex + "' out of range.");
                }
                minorityClassIndex = idx - 1; // Convert 1-based to 0-based
                minSize = data.attributeStats(classIndex).nominalCounts[minorityClassIndex];
                if (minSize == 0) {
                     throw new Exception("Specified class index " + m_ClassValueIndex + " has zero instances.");
                }
            } catch (NumberFormatException e) {
                throw new Exception("Invalid class value index: " + m_ClassValueIndex);
            }
        }
        System.err.println("FRSMOTE: Targeting minority class index: " + minorityClassIndex + " ('" + classAttr.value(minorityClassIndex) + "') with " + minSize + " instances.");

        // --- 2. Push original instances ---
        for (int i = 0; i < numInstances; i++) {
            push(data.instance(i)); // Push original data first
        }

        if (minSize <= 1) {
             System.err.println("Warning: Minority class size is <= 1. Cannot apply FRSMOTE.");
             return; // Nothing more to do
        }

        // --- 3. Prepare FRS Components & Data ---
        setupFuzzyComponents(data); // Pass data for similarity setup if needed

        // Normalize data (excluding class)
        Instances normalizedData = normalizeData(data);

        // Calculate Similarity Matrix
        System.err.println("FRSMOTE: Calculating similarity matrix...");
        m_SimilarityMatrix = calculateSimilarityMatrix(normalizedData);

        // Calculate Positive Region Membership
        System.err.println("FRSMOTE: Calculating positive region membership...");
        m_PosMembership = calculatePositiveRegionMembership(normalizedData, m_SimilarityMatrix);

        // --- 4. Filter Minority Instances and Select Candidates ---
        ArrayList<Integer> minorityIndices = new ArrayList<>();
        for (int i = 0; i < numInstances; i++) {
            if ((int) data.instance(i).classValue() == minorityClassIndex) {
                minorityIndices.add(i);
            }
        }

        // Selectable candidates (POS > 0)
        ArrayList<Pair> p1Candidates = new ArrayList<>(); // Pair: (index, pos_membership)
        ArrayList<Integer> selectableIndices = new ArrayList<>(); // Store indices of selectable points
        for (int idx : minorityIndices) {
            if (m_PosMembership[idx] > EPSILON) {
                p1Candidates.add(new Pair(idx, m_PosMembership[idx]));
                selectableIndices.add(idx);
            }
        }

        if (p1Candidates.size() < 2) {
            System.err.println("Warning: Fewer than 2 minority instances with POS membership > 0 found (" + p1Candidates.size() + "). SMOTE generation might be limited or impossible. Consider FRS parameters.");
            // If we have at least 2 original minority points, we might proceed using them,
            // but the weighted selection won't work well. Let's allow it but use a fallback.
             if (minorityIndices.size() >= 2 && p1Candidates.isEmpty()) {
                 System.err.println("Using all minority instances for selection (ignoring POS weights).");
                 for(int idx : minorityIndices) {
                     p1Candidates.add(new Pair(idx, 1.0)); // Assign dummy weight 1.0
                     selectableIndices.add(idx);
                 }
             } else if (p1Candidates.size() < 2) {
                 return; // Cannot proceed if < 2 selectable overall
             }
        }

        // --- 5. Prepare K-NN Structure (using fuzzy similarity) ---
        // We need to find K nearest neighbors based on the m_SimilarityMatrix
        // for each *selectable* minority instance, considering only other *selectable* minority instances.
        Map<Integer, List<Pair>> fuzzyNeighbors = findFuzzyNeighbors(selectableIndices, m_SimilarityMatrix, m_NearestNeighbors);

        // --- 6. Generate Synthetic Samples ---
        System.err.println("FRSMOTE: Generating synthetic samples...");
        int numSynthetic = (int) Math.round(m_Percentage / 100.0 * minorityIndices.size());
        if (numSynthetic == 0 && m_Percentage > 0) {
             System.err.println("Warning: Calculated 0 synthetic samples to generate, although percentage > 0.");
        }

        Random random = new Random(m_RandomSeed);
        int generatedCount = 0;

        for (int i = 0; i < numSynthetic; i++) {
            // Select base point p1 using weighted random choice
            Pair p1Pair = weightedRandomChoice(p1Candidates, random);
            if (p1Pair == null) continue; // Should not happen if candidates exist
            int p1Index = p1Pair.index;
            Instance p1Instance = data.instance(p1Index);

            // Get neighbors of p1 from the precalculated map
            List<Pair> p1Neighbors = fuzzyNeighbors.get(p1Index);
            if (p1Neighbors == null || p1Neighbors.isEmpty()) {
                // This can happen if K is >= number of selectable candidates
                // Or if p1 was not in selectableIndices (shouldn't happen with current logic)
                // Fallback: randomly pick another selectable point? Or skip? Let's skip.
                // System.err.println("Skipping sample generation for instance " + p1Index + ": No fuzzy neighbors found.");
                i--; // Try again to reach target count
                continue;
            }

            // Select neighbor p2 randomly from the K fuzzy neighbors
            int nnIndexInList = random.nextInt(p1Neighbors.size());
            int p2Index = p1Neighbors.get(nnIndexInList).index;
            Instance p2Instance = data.instance(p2Index);

            // Calculate lambda
            double lambda;
            if (m_BiasInterpolation) {
                double p1Pos = m_PosMembership[p1Index];
                double p2Pos = m_PosMembership[p2Index];
                double denominator = p1Pos + p2Pos;
                lambda = (denominator < EPSILON) ? 0.5 : (p2Pos / denominator); // Bias towards higher POS
                lambda = Math.max(0.0, Math.min(1.0, lambda)); // Clamp
            } else {
                lambda = random.nextDouble();
            }

            // Create synthetic instance values
            double[] values = new double[data.numAttributes()];
            for (int j = 0; j < data.numAttributes(); j++) {
                Attribute attr = data.attribute(j);
                if (j == classIndex) {
                    values[j] = minorityClassIndex;
                } else {
                    double val1 = p1Instance.value(j);
                    double val2 = p2Instance.value(j);
                    if (p1Instance.isMissing(j) || p2Instance.isMissing(j)) {
                        // Handle missing values - maybe copy from non-missing parent? Or set as missing?
                        // Simplest: If either is missing, result is missing (Weka handles DenseInstance missing)
                        values[j] = Utils.missingValue();
                    } else if (attr.isNumeric()) {
                        values[j] = val1 + lambda * (val2 - val1);
                    } else if (attr.isDate()) {
                        long date1 = (long) val1;
                        long date2 = (long) val2;
                        values[j] = (double) (date1 + (long) (lambda * (date2 - date1)));
                    } else { // Nominal
                        // Use majority vote between parents and their K neighbors (like original SMOTE)
                        // This requires finding neighbors again which is slow, OR use precomputed ones.
                        // Simpler approach: Randomly pick parent or neighbor's value, or just p1/p2's value.
                        // Let's use the SMOTE majority vote approach for nominals, gathering relevant neighbors.
                        // Find SMOTE's view of neighbors for nominal generation
                        values[j] = generateNominalValue(p1Instance, p2Instance, p1Neighbors, attr, random);
                    }
                }
            }
            // Create and push synthetic instance
            Instance synthetic = new DenseInstance(1.0, values);
            // Important: Set dataset reference for the new instance
            synthetic.setDataset(data); // Use the original structure reference
            push(synthetic);
            generatedCount++;
        }
         System.err.println("FRSMOTE: Generated " + generatedCount + " synthetic instances.");
    }


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
                    double[] weights = OWAUtils.owaWeightsLinear(dissimilarities.size());
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

     /** Find K nearest neighbors based on precomputed fuzzy similarity matrix */
     protected Map<Integer, List<Pair>> findFuzzyNeighbors(List<Integer> targetIndices, double[][] simMatrix, int k) {
         Map<Integer, List<Pair>> neighborsMap = new HashMap<>();
         if (targetIndices == null || targetIndices.isEmpty()) return neighborsMap;

         // Create a list of pairs (index, similarity) for efficient sorting
         List<Pair> similarities = new ArrayList<>();

         for (int i : targetIndices) {
             similarities.clear();
             for (int j : targetIndices) {
                 if (i == j) continue;
                 similarities.add(new Pair(j, simMatrix[i][j])); // Store index j and similarity Sim(i,j)
             }

             // Sort by similarity descending (highest first)
             Collections.sort(similarities, Collections.reverseOrder());

             // Get top K neighbors
             List<Pair> topK = new ArrayList<>();
             int count = 0;
             for (Pair p : similarities) {
                 if (count < k) {
                     topK.add(p); // Add pair (neighbor_index, similarity_score)
                     count++;
                 } else {
                     break;
                 }
             }
             neighborsMap.put(i, topK);
         }
         return neighborsMap;
     }


    /** Helper to generate nominal value using SMOTE's majority vote */
    protected double generateNominalValue(Instance p1, Instance p2, List<Pair> p1Neighbors, Attribute attr, Random random) {
        // Count votes for nominal value from p1 and its fuzzy neighbors
        // Note: SMOTE originally uses KNN based on combined distance, here we use fuzzy neighbors
        int[] valueCounts = new int[attr.numValues()];
        valueCounts[(int) p1.value(attr)]++; // Count p1
        for (Pair neighborPair : p1Neighbors) {
             Instance neighborInstance = getInputFormat().instance(neighborPair.index); // Need original Instances reference
             if (!neighborInstance.isMissing(attr.index())) {
                  valueCounts[(int) neighborInstance.value(attr)]++;
             }
        }

        int maxCount = -1;
        ArrayList<Integer> majorityIndices = new ArrayList<>();
        for(int v=0; v<valueCounts.length; v++) {
            if (valueCounts[v] > maxCount) {
                maxCount = valueCounts[v];
                majorityIndices.clear();
                majorityIndices.add(v);
            } else if (valueCounts[v] == maxCount) {
                 majorityIndices.add(v);
            }
        }
        // If tie, randomly pick one of the majority values
        return (double) majorityIndices.get(random.nextInt(majorityIndices.size()));
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

    /** Simple Pair class to hold index and value (e.g., POS membership or similarity) */
    protected static class Pair implements Comparable<Pair> {
        int index;
        double value;

        Pair(int index, double value) {
            this.index = index;
            this.value = value;
        }

        // Compare based on value (descending for sorting by score/similarity)
        @Override
        public int compareTo(Pair other) {
            return Double.compare(this.value, other.value);
        }
    }

     /** OWA Helper Utilties */
     protected static class OWAUtils {
          public static double[] owaWeightsLinear(int n) {
               if (n <= 0) return new double[0];
               if (n == 1) return new double[]{1.0};
               double[] weights = new double[n];
               double denominator = (double) n * (n + 1) / 2.0;
               if (denominator < EPSILON) return Arrays.stream(weights).map(w -> 1.0/n).toArray(); // Uniform fallback
               for (int i = 0; i < n; i++) {
                    weights[i] = (double) (n - i) / denominator; // Weight for i-th element in *sorted* list (index i)
               }
               return weights;
          }
     }

    /**
     * Main method for running this filter.
     * @param args should contain arguments to the filter: use -h for help
     */
    public static void main(String[] args) {
        runFilter(new FRSMOTE(), args);
    }
}