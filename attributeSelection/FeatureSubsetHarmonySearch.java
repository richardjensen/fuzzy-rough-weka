package weka.attributeSelection;

import weka.core.*;

import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: rrd09
 * Date: 21-Oct-2009
 * Time: 12:44:47
 * To change this template use File | Settings | File Templates.
 */
public class FeatureSubsetHarmonySearch extends ASSearch
        implements StartSetHandler, OptionHandler, TechnicalInformationHandler {

    private FeatureSubsetHarmonyMemory featureSubsetHarmonyMemory;// = new FeatureSubsetHarmonyMemory(new FeatureSubsetComparator(), 15);
    //private HarmonyMemory featureVoteHarmonyMemory;
    public int memorySize = 50;

    public int iteration = 1000;
    
    /**
     * for serialization
     */
    static final long serialVersionUID = 7479392617377425484L;

    /**
     * holds a starting set as an array of attributes.
     */
    private int[] m_starting;

    /**
     * holds the start set as a range
     */
    private Range m_startRange;

    /**
     * the best feature set found during the search
     */
    private BitSet m_bestGroup;

    /**
     * the merit of the best subset found
     */
    private double m_bestMerit;

    /**
     * only accept a feature set as being "better" than the best if its
     * merit is better or equal to the best, and it contains fewer
     * features than the best (this allows LVF to be implimented).
     */
    private boolean m_onlyConsiderBetterAndSmaller;

    /**
     * does the data have a class
     */
    private boolean m_hasClass;

    /**
     * holds the class index
     */
    private int m_classIndex;

    /**
     * number of attributes in the data
     */
    private int m_numAttribs;

    /**
     * seed for random number generation
     */
    private int m_seed;

    /**
     * percentage of the search space to consider
     */
    private double m_searchSize;

    /**
     * the number of iterations performed
     */
    private int m_iterations;

    /**
     * random number object
     */
    private Random m_random;

    /**
     * output new best subsets as the search progresses
     */
    private boolean m_verbose;

    /**
     * Returns a string describing this search method
     *
     * @return a description of the search suitable for
     *         displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "RandomSearch : \n\nPerforms a Random search in "
                + "the space of attribute subsets. If no start set is supplied, Random "
                + "search starts from a random point and reports the best subset found. "
                + "If a start set is supplied, Random searches randomly for subsets "
                + "that are as good or better than the start point with the same or "
                + "or fewer attributes. Using RandomSearch in conjunction with a start "
                + "set containing all attributes equates to the LVF algorithm of Liu "
                + "and Setiono (ICML-96).\n\n"
                + "For more information see:\n\n"
                + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing
     * detailed information about the technical background of this class,
     * e.g., paper reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
        result.setValue(TechnicalInformation.Field.AUTHOR, "H. Liu and R. Setiono");
        result.setValue(TechnicalInformation.Field.TITLE, "A probabilistic approach to feature selection - A filter solution");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "13th International Conference on Machine Learning");
        result.setValue(TechnicalInformation.Field.YEAR, "1996");
        result.setValue(TechnicalInformation.Field.PAGES, "319-327");

        return result;
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }

    /**
     * Constructor
     */
    public FeatureSubsetHarmonySearch() {
        resetOptions();
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {
        Vector newVector = new Vector(5);

        newVector.addElement(new Option("\tSpecify a starting set of attributes."
                + "\n\tEg. 1,3,5-7."
                + "\n\tIf a start point is supplied,"
                + "\n\trandom search evaluates the start"
                + "\n\tpoint and then randomly looks for"
                + "\n\tsubsets that are as good as or better"
                + "\n\tthan the start point with the same"
                + "\n\tor lower cardinality."
                , "P", 1
                , "-P <start set>"));

        newVector.addElement(new Option("\tPercent of search space to consider."
                + "\n\t(default = 25%)."
                , "F", 1
                , "-F <percent> "));
        newVector.addElement(new Option("\tOutput subsets as the search progresses."
                + "\n\t(default = false)."
                , "V", 0
                , "-V"));
        newVector.addElement(new Option("\tTotal Iterations."
                + "\n\t(default = false)."
                , "I", 1000
                , "-I"));

        newVector.addElement(new Option("\tMemory Size."
                + "\n\t(default = false)."
                , "M", 50
                , "-M"));
        return newVector.elements();
    }

    public int getMemorySize() {
        return memorySize;
    }

    public void setMemorySize(int memorySize) {
        this.memorySize = memorySize;
    }

    /**
     * Parses a given list of options. <p/>
     * <p/>
     * <!-- options-start -->
     * Valid options are: <p/>
     * <p/>
     * <pre> -P &lt;start set&gt;
     *  Specify a starting set of attributes.
     *  Eg. 1,3,5-7.
     *  If a start point is supplied,
     *  random search evaluates the start
     *  point and then randomly looks for
     *  subsets that are as good as or better
     *  than the start point with the same
     *  or lower cardinality.</pre>
     * <p/>
     * <pre> -F &lt;percent&gt;
     *  Percent of search space to consider.
     *  (default = 25%).</pre>
     * <p/>
     * <pre> -V
     *  Output subsets as the search progresses.
     *  (default = false).</pre>
     * <p/>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options)
            throws Exception {
        String optionString;
        resetOptions();

        optionString = Utils.getOption('P', options);
        if (optionString.length() != 0) {
            setStartSet(optionString);
        }

        optionString = Utils.getOption('F', options);
        if (optionString.length() != 0) {
            setSearchPercent(Double.valueOf(optionString));
        }

        optionString = Utils.getOption('I', options);
        if (optionString.length() != 0) {
            setIteration(Integer.valueOf(optionString));
        }

        optionString = Utils.getOption('M', options);
        if (optionString.length() != 0) {
            setMemorySize(Integer.valueOf(optionString));
        }

        setVerbose(Utils.getFlag('V', options));
    }

    /**
     * Gets the current settings of RandomSearch.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions() {
        String[] options = new String[9];
        int current = 0;

        if (m_verbose) {
            options[current++] = "-V";
        }
        
        options[current++] = "-I";
        options[current++] = "" + getIteration();
        
        options[current++] = "-M";
        options[current++] = "" + getMemorySize();

        if (!(getStartSet().equals(""))) {
            options[current++] = "-P";
            options[current++] = "" + startSetToString();
        }

        options[current++] = "-F";
        options[current++] = "" + getSearchPercent();

        while (current < options.length) {
            options[current++] = "";
        }

        return options;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     *         displaying in the explorer/experimenter gui
     */
    public String startSetTipText() {
        return "Set the start point for the search. This is specified as a comma "
                + "seperated list off attribute indexes starting at 1. It can include "
                + "ranges. Eg. 1,2,5-9,17. If specified, Random searches for subsets "
                + "of attributes that are as good as or better than the start set with "
                + "the same or lower cardinality.";
    }

    /**
     * Sets a starting set of attributes for the search. It is the
     * search method's responsibility to report this start set (if any)
     * in its toString() method.
     *
     * @param startSet a string containing a list of attributes (and or ranges),
     *                 eg. 1,2,6,10-15. "" indicates no start point.
     *                 If a start point is supplied, random search evaluates the
     *                 start point and then looks for subsets that are as good as or better
     *                 than the start point with the same or lower cardinality.
     * @throws Exception if start set can't be set.
     */
    public void setStartSet(String startSet) throws Exception {
        m_startRange.setRanges(startSet);
    }

    /**
     * Returns a list of attributes (and or attribute ranges) as a String
     *
     * @return a list of attributes (and or attribute ranges)
     */
    public String getStartSet() {
        return m_startRange.getRanges();
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     *         displaying in the explorer/experimenter gui
     */
    public String verboseTipText() {
        return "Print progress information. Sends progress info to the terminal "
                + "as the search progresses.";
    }

    /**
     * set whether or not to output new best subsets as the search proceeds
     *
     * @param v true if output is to be verbose
     */
    public void setVerbose(boolean v) {
        m_verbose = v;
    }

    /**
     * get whether or not output is verbose
     *
     * @return true if output is set to verbose
     */
    public boolean getVerbose() {
        return m_verbose;
    }

    /**
     * Returns the tip text for this property
     *
     * @return tip text for this property suitable for
     *         displaying in the explorer/experimenter gui
     */
    public String searchPercentTipText() {
        return "Percentage of the search space to explore.";
    }

    /**
     * set the percentage of the search space to consider
     *
     * @param p percent of the search space ( 0 < p <= 100)
     */
    public void setSearchPercent(double p) {
        p = Math.abs(p);
        if (p == 0) {
            p = 25;
        }

        if (p > 100.0) {
            p = 100;
        }

        m_searchSize = (p / 100.0);
    }

    /**
     * get the percentage of the search space to consider
     *
     * @return the percent of the search space explored
     */
    public double getSearchPercent() {
        return m_searchSize * 100;
    }

    /**
     * converts the array of starting attributes to a string. This is
     * used by getOptions to return the actual attributes specified
     * as the starting set. This is better than using m_startRanges.getRanges()
     * as the same start set can be specified in different ways from the
     * command line---eg 1,2,3 == 1-3. This is to ensure that stuff that
     * is stored in a database is comparable.
     *
     * @return a comma seperated list of individual attribute numbers as a String
     */
    private String startSetToString() {
        StringBuffer FString = new StringBuffer();
        boolean didPrint;

        if (m_starting == null) {
            return getStartSet();
        }

        for (int i = 0; i < m_starting.length; i++) {
            didPrint = false;

            if ((m_hasClass == false) ||
                    (m_hasClass == true && i != m_classIndex)) {
                FString.append((m_starting[i] + 1));
                didPrint = true;
            }

            if (i == (m_starting.length - 1)) {
                FString.append("");
            } else {
                if (didPrint) {
                    FString.append(",");
                }
            }
        }

        return FString.toString();
    }

    /**
     * prints a description of the search
     *
     * @return a description of the search as a string
     */
    public String toString() {
        StringBuffer text = new StringBuffer();

        text.append("\tRandom search.\n\tStart set: ");
        if (m_starting == null) {
            text.append("no attributes\n");
        } else {
            text.append(startSetToString() + "\n");
        }
        text.append("\tNumber of iterations: " + m_iterations + " ("
                + (m_searchSize * 100.0) + "% of the search space)\n");
        text.append("\tMerit of best subset found: "
                + Utils.doubleToString(Math.abs(m_bestMerit), 8, 3) + "\n");

        return text.toString();
    }

    public void initialiseHarmonyMemory(int size) {

    }

    /**
     * Searches the attribute subset space randomly.
     *
     * @param ASEval the attribute evaluator to guide the search
     * @param data   the training instances.
     * @return an array (not necessarily ordered) of selected attribute indexes
     * @throws Exception if the search can't be completed
     */
    public int[] search(ASEvaluation ASEval, Instances data)
            throws Exception {

        //System.out.println("hi");
        featureSubsetHarmonyMemory = new FeatureSubsetHarmonyMemory(memorySize);
        //featureVoteHarmonyMemory = new HarmonyMemory(memorySize);

        double best_merit;
        int sizeOfBest = m_numAttribs;
        FeatureSubset temp;
        m_bestGroup = new FeatureSubset(m_numAttribs);

        m_onlyConsiderBetterAndSmaller = false;
        if (!(ASEval instanceof SubsetEvaluator)) {
            throw new Exception(ASEval.getClass().getName()
                    + " is not a "
                    + "Subset evaluator!");
        }

        m_random = new Random(m_seed);

        if (ASEval instanceof UnsupervisedSubsetEvaluator) {
            m_hasClass = false;
        } else {
            m_hasClass = true;
            m_classIndex = data.classIndex();
        }

        SubsetEvaluator ASEvaluator = (SubsetEvaluator) ASEval;
        m_numAttribs = data.numAttributes();

        m_startRange.setUpper(m_numAttribs - 1);
        if (!(getStartSet().equals(""))) {
            m_starting = m_startRange.getSelection();
        }

        // If a starting subset has been supplied, then initialise the bitset
        if (m_starting != null) {
            for (int i = 0; i < m_starting.length; i++) {
                if ((m_starting[i]) != m_classIndex) {
                    m_bestGroup.set(m_starting[i]);
                }
            }
            m_onlyConsiderBetterAndSmaller = true;
            best_merit = ASEvaluator.evaluateSubset(m_bestGroup);
            sizeOfBest = countFeatures(m_bestGroup);
        } else {
            // do initial random subset
            m_bestGroup = generateRandomSubset();
            best_merit = ASEvaluator.evaluateSubset(m_bestGroup);
        }

        if (m_verbose) {
            System.out.println("Initial subset ("
                    + Utils.doubleToString(Math.
                    abs(best_merit), 8, 5)
                    + "): " + printSubset(m_bestGroup));
        }

        int i;
        if (m_hasClass) {
            i = m_numAttribs - 1;
        } else {
            i = m_numAttribs;
        }
        m_iterations = (int) ((m_searchSize * Math.pow(2, i)));

        int tempSize;
        double tempMerit;

        //populate memory
        /*for (int k = 0; k < memorySize; k++) {
            FeatureSubset fs = generateRandomSubset();
            fs.setMerit(ASEvaluator.evaluateSubset(fs));
            //System.out.println("Adding New FeatureSubset " + fs.toString() + " - " + fs.getMerit() + " - " + featureSubsetHarmonyMemory.add(fs));
        }*/

        /*for (int kk = 0; kk < m_numAttribs; kk ++) {
                //System.out.println("New NoteDomain " + kk);
                featureVoteHarmonyMemory.getNoteDomains().add(new NoteDomain(m_numAttribs));
                //System.out.println("New NoteDomain " + kk + " done");
            }

        while (featureVoteHarmonyMemory.size() < memorySize) {
            Vote fs = generateRandomFeatureVote();
            System.out.println(fs.toString());
            BitSet bs = fs.toBitSet();
            fs.setMerit(ASEvaluator.evaluateSubset(bs));
            fs.setCardinality(bs.cardinality());
            System.out.println("Adding New Vote " + bs.toString() + " - " + fs.getMerit() + " - " + fs.cardinality() + " - " + featureVoteHarmonyMemory.add(fs));
        }*/

        int lastAdded = 0;
        int optimalAdded = 0;
        int addedCount = 0;
        int rejectedCount = 0;
        int initialSeedCount = 0;

        while (featureSubsetHarmonyMemory.size() < memorySize) {
            FeatureSubset fs = generateRandomSubset();
            fs.setMerit(ASEvaluator.evaluateSubset(fs));            
            featureSubsetHarmonyMemory.add(fs);
            initialSeedCount = initialSeedCount + 1;
            //System.out.println("Adding New Vote " + bs.toString() + " - " + fs.getMerit() + " - " + fs.cardinality() + " - " + featureVoteHarmonyMemory.add(fs));
        }

        // main loop
        System.out.println("Initiated Search with " + iteration + " iterations");
        for (int m = 0; m < iteration; m++) {


/*            temp = generateRandomSubset();
            if (m_onlyConsiderBetterAndSmaller) {
                tempSize = countFeatures(temp);
                if (tempSize <= sizeOfBest) {
                    tempMerit = ASEvaluator.evaluateSubset(temp);
                    if (tempMerit >= best_merit) {
                        sizeOfBest = tempSize;
                        m_bestGroup = temp;
                        best_merit = tempMerit;
                        if (m_verbose) {
                            System.out.print("New best subset ("
                                    + Utils.doubleToString(Math.
                                    abs(best_merit), 8, 5)
                                    + "): " + printSubset(m_bestGroup) + " :");
                            System.out.println(Utils.
                                    doubleToString((((double) i) /
                                            ((double) m_iterations) *
                                            100.0), 5, 1)
                                    + "% done");
                        }
                    }
                }
            } else {
                tempMerit = ASEvaluator.evaluateSubset(temp);
                if (tempMerit > best_merit) {
                    m_bestGroup = temp;
                    best_merit = tempMerit;
                    if (m_verbose) {
                        System.out.print("New best subset ("
                                + Utils.doubleToString(Math.abs(best_merit), 8, 5)
                                + "): " + printSubset(m_bestGroup) + " :");
                        System.out.println(Utils.
                                doubleToString((((double) i) /
                                        ((double) m_iterations)
                                        * 100.0), 5, 1)
                                + "% done");
                    }
                }
            }*/


            FeatureSubset fs = featureSubsetHarmonyMemory.newHarmony();
            fs.setMerit(ASEvaluator.evaluateSubset(fs));

            Boolean added = featureSubsetHarmonyMemory.add(fs);
            if (added) {
                //System.out.println("Added Vote " + fs.toString() + " - " + fs.getMerit());
                addedCount = addedCount + 1;
                lastAdded = m;
                if (featureSubsetHarmonyMemory.comparator().compare(featureSubsetHarmonyMemory.last(), fs) == 0) {
                    optimalAdded = m;
                }
            } else {
                rejectedCount = rejectedCount + 1;
                //System.out.println("Rejected Vote " + fs.toString() + " - " + fs.getMerit());
            }
            //System.out.println("Adding New FeatureSubset " + fs.toString() + " - " + fs.getMerit() + " - " + featureSubsetHarmonyMemory.add(fs));


        }
        int rank = 1;
        for (FeatureSubset fts : featureSubsetHarmonyMemory) {
            System.out.println(rank + fts.toString() + " - " + fts.getMerit());
            rank++;
        }
        System.out.println("Last Addition on the " + lastAdded + "-th Iteration");
        System.out.println("Optimal Found on the " + optimalAdded + "-th Iteration");
        System.out.println("Initial Seed " + initialSeedCount + " - Added " + addedCount + " - Rejected " + rejectedCount);
        m_bestMerit = featureSubsetHarmonyMemory.last().getMerit();
        return attributeList(featureSubsetHarmonyMemory.last());


        /*Vote fs = featureVoteHarmonyMemory.newHarmony();
        BitSet bs = fs.toBitSet();
        fs.setMerit(ASEvaluator.evaluateSubset(bs));
        fs.setCardinality(bs.cardinality());
        Boolean added = featureVoteHarmonyMemory.add(fs);
        if (added) {
            System.out.println("Added Vote " + bs.toString() + " - " + fs.getMerit());
            lastAdded = m;
        }
        else {
            System.out.println("Rejected Vote " + bs.toString() + " - " + fs.getMerit());
        }

        }
        int rank = 1;
        for (Vote fv : featureVoteHarmonyMemory) {
            BitSet bss = fv.toBitSet();
            System.out.println(rank + bss.toString() + " - " + fv.getMerit());
            rank ++;
        }


        System.out.println("Last Addition on the " + lastAdded + "-th Iteration");
        m_bestMerit = ASEvaluator.evaluateSubset(featureVoteHarmonyMemory.last().toBitSet());
        return attributeList(featureVoteHarmonyMemory.last().toBitSet());*/


        //m_bestMerit = best_merit;
        //return attributeList(m_bestGroup);
    }

    /**
     * prints a subset as a series of attribute numbers
     *
     * @param temp the subset to print
     * @return a subset as a String of attribute numbers
     */
    private String printSubset(BitSet temp) {
        StringBuffer text = new StringBuffer();

        for (int j = 0; j < m_numAttribs; j++) {
            if (temp.get(j)) {
                text.append((j + 1) + " ");
            }
        }
        return text.toString();
    }

    /**
     * converts a BitSet into a list of attribute indexes
     *
     * @param group the BitSet to convert
     * @return an array of attribute indexes
     */
    private int[] attributeList(BitSet group) {
        int count = 0;

        // count how many were selected
        for (int i = 0; i < m_numAttribs; i++) {
            if (group.get(i)) {
                count++;
            }
        }

        int[] list = new int[count];
        count = 0;

        for (int i = 0; i < m_numAttribs; i++) {
            if (group.get(i)) {
                list[count++] = i;
            }
        }

        return list;
    }

    /**
     * generates a random subset
     *
     * @return a random subset as a BitSet
     */
    private FeatureSubset generateRandomSubset() {
        FeatureSubset temp = new FeatureSubset(m_numAttribs - 1);
        double r;

        for (int i = 0; i < m_numAttribs - 1; i++) {
            r = m_random.nextDouble();
            if (r <= 0.5) {
                if (m_hasClass && i == m_classIndex) {
                } else {
                    temp.set(i);
                }
            }
        }
        return temp;
    }

    private FeatureVote generateRandomFeatureVote() {
        FeatureVote fs = new FeatureVote(m_numAttribs - 1);
        for (int i = 0; i < fs.getSize(); i++) {
            if (m_random.nextDouble() < 0.5) {
                fs.set(i, m_random.nextInt(m_numAttribs));
            } else {
                fs.set(i, m_numAttribs - 1);
            }
        }
        return fs;
    }

    /**
     * counts the number of features in a subset
     *
     * @param featureSet the feature set for which to count the features
     * @return the number of features in the subset
     */
    private int countFeatures(BitSet featureSet) {
        return featureSet.cardinality();
        /*int count = 0;
       for (int i=0;i<m_numAttribs;i++) {
         if (featureSet.get(i)) {
       count++;
         }
       }
       return count;*/
    }

    /**
     * resets to defaults
     */
    private void resetOptions() {
        m_starting = null;
        m_startRange = new Range();
        m_searchSize = 0.25;
        m_seed = 1;
        m_onlyConsiderBetterAndSmaller = false;
        m_verbose = false;
    }

    protected class FeatureSubset extends BitSet {
        private double merit;

        public FeatureSubset(double merit) {
            this.merit = merit;
        }

        public FeatureSubset(int nbits, double merit) {
            super(nbits);
            this.merit = merit;
        }

        public FeatureSubset() {
            super();
        }

        public FeatureSubset(int nbits) {
            super(nbits);
        }

        public double getMerit() {
            return merit;
        }

        public void setMerit(double merit) {
            this.merit = merit;
        }
    }

    protected class FeatureVote {
        private double merit;
        private int cardinality;
        private int[] featureVotes;

        public int getSize() {
            return featureVotes.length;
        }

        public FeatureVote(int size) {
            featureVotes = new int[size];
        }

        public BitSet toBitSet() {
            //System.out.println("blah");
            BitSet bs = new BitSet(m_numAttribs - 1);
            for (int i = 0; i < featureVotes.length; i++) {
                if ((featureVotes[i] >= 0) && (featureVotes[i] < m_numAttribs - 1)) {
                    //bs.set(m_numAttribs-1 - featureVotes[i]);
                    bs.set(featureVotes[i]);
                }
            }
            //System.out.println(bs.toString());
            return bs;
        }

        public double getMerit() {
            return merit;
        }

        public void setMerit(double merit) {
            this.merit = merit;
        }

        public int cardinality() {
            return cardinality;
        }

        public void setCardinality(int cardinality) {
            this.cardinality = cardinality;
        }

        public int[] getFeatureVotes() {
            return featureVotes;
        }

        public void setFeatureVotes(int[] featureVotes) {
            this.featureVotes = featureVotes;
        }

        public void set(int index, int value) {
            featureVotes[index] = value;
        }

        public int get(int index) {
            //System.out.println(featureVotes[index] + " for " + index);
            return featureVotes[index];
        }

        public String toString() {
            String s = "";
            for (int i : featureVotes) {
                s = s + " " + i;
            }
            return s;
        }
    }

    protected class FeatureVoteHarmonyMemory extends TreeSet<FeatureVote> {
        private int memorySize;
        private ArrayList<NoteDomain> noteDomains;

        public FeatureVoteHarmonyMemory() {
        }

        public ArrayList<NoteDomain> getNoteDomains() {
            return noteDomains;
        }

        public void setNoteDomains(ArrayList<NoteDomain> noteDomains) {
            this.noteDomains = noteDomains;
        }

        public FeatureVoteHarmonyMemory(int memorySize) {
            super(new FeatureVoteComparator());
            //System.out.println("New HarmonyMemory");
            this.memorySize = memorySize;
            noteDomains = new ArrayList<NoteDomain>();
            //System.out.println("New NoteDomains");

        }

        public int getMemorySize() {
            return memorySize;
        }//true if added and false otherwise

        public void setMemorySize(int memorySize) {
            this.memorySize = memorySize;
        }

        public boolean add(FeatureVote e) {

            if (super.contains(e)) return false;
            if (super.size() < memorySize) {
                for (int i = 0; i < e.getSize(); i++) {
                    //System.out.println("Adding " + e.get(i) + " to " + i);
                    noteDomains.get(i).addNote(e.get(i));
                }
                super.add(e);
                return true;
            }
            if (super.comparator().compare(e, super.first()) >= 0) {
                for (int i = 0; i < e.getSize(); i++) {
                    noteDomains.get(i).replaceNote(this.first().get(i), e.get(i));
                }
                super.remove(this.first());
                super.add(e);
                return true;
            }
            return false;
        }

        public FeatureVote getLeastHarmony() {
            int currentBestCount = m_numAttribs;
            FeatureVote currentBest = null;
            for (FeatureVote fs : this) {
                if (fs.cardinality() < currentBestCount) {
                    currentBest = fs;
                }
            }
            return currentBest;
        }

        public FeatureVote newHarmony() {
            FeatureVote newFS = new FeatureVote(m_numAttribs - 1);
            for (int i = 0; i < m_numAttribs - 1; i++) {
                if (m_random.nextDouble() < 0.7) {
                    if (m_random.nextDouble() < 0.25) {
                        newFS.set(i, noteDomains.get(i).pickNote());
                    } else {
                        newFS.set(i, m_numAttribs - 1);
                    }
                } else {
                    if (m_random.nextDouble() < 0.5) {
                        if (m_random.nextDouble() < 0.5) {
                            newFS.set(i, noteDomains.get(i).pickNote() + 1);
                        } else {
                            newFS.set(i, noteDomains.get(i).pickNote() - 1);
                        }
                    } else {
                        newFS.set(i, m_random.nextInt(m_numAttribs));
                    }

                }
            }
            return newFS;
        }

    }

    protected class FeatureSubsetHarmonyMemory extends TreeSet<FeatureSubset> {
        private int memorySize;

        public FeatureSubsetHarmonyMemory() {
        }

        public FeatureSubsetHarmonyMemory(int memorySize) {
            super(new FeatureSubsetComparator());
            this.memorySize = memorySize;
        }

        public int getMemorySize() {
            return memorySize;
        }//true if added and false otherwise

        public void setMemorySize(int memorySize) {
            this.memorySize = memorySize;
        }

        public boolean add(FeatureSubset e) {
            if (super.contains(e)) return false;
            if (super.size() < memorySize) {
                super.add(e);
                return true;
            }
            if (super.comparator().compare(e, super.first()) >= 0) {
                super.remove(this.first());
                super.add(e);
                return true;
            }
            return false;
        }

        public FeatureSubset getLeastHarmony() {
            int currentBestCount = m_numAttribs;
            FeatureSubset currentBest = null;
            for (FeatureSubset fs : this) {
                if (fs.cardinality() < currentBestCount) {
                    currentBest = fs;
                }
            }
            return currentBest;
        }

        public FeatureSubset newHarmony() {
            FeatureSubset newFS = new FeatureSubset(m_numAttribs - 1);
            for (int i = 0; i < m_numAttribs - 1; i++) {
                if (m_random.nextDouble() > 0.9) {
                    /*if (m_random.nextDouble() > 0.5) {
                        newFS.set(i, true);
                    }
                    else {
                        newFS.set(i, false);
                    }*/
                    newFS.set(i, !chooseNote(m_random.nextInt(memorySize), i));
                } else {
                    newFS.set(i, chooseNote(m_random.nextInt(memorySize), i));
                }
            }
            return newFS;
        }

        private boolean chooseNote(int harmonyIndex, int noteIndex) {
            int j = 0;
            for (FeatureSubset fs : this) {
                if (j == harmonyIndex) {
                    return fs.get(noteIndex);
                }
                j = j + 1;
            }
            return false;
        }
    }

    public class NoteDomain {

        private int[] notes;
        private int cardinality;

        public NoteDomain() {
            notes = new int[m_numAttribs];
            cardinality = 0;
        }

        public NoteDomain(int size) {
            notes = new int[size];
            cardinality = 0;
            //System.out.println("New NoteDomain Size = " + size);
        }

        public int getSize() {
            return notes.length;
        }

        public void addNote(int index) {
            //System.out.println("Size = " + notes.length + " - " + notes[index]);
            notes[index] = notes[index] + 1;
            cardinality = cardinality + 1;
        }

        public void replaceNote(int oldIndex, int newIndex) {
            notes[oldIndex] = (notes[oldIndex] <= 0) ? 0 : (notes[oldIndex] - 1);
            notes[newIndex] = notes[newIndex] + 1;
        }

        public void removeNote(int index) {
            notes[index] = (notes[index] <= 0) ? 0 : (notes[index] - 1);
            cardinality = (cardinality <= 0) ? 0 : cardinality - 1;
        }

        public int pickNote() {
            int index = m_random.nextInt(cardinality + 1);
            for (int i = 0; i < notes.length; i++) {
                if (notes[i] >= index) return i;
                index = index - notes[i];
            }
            return 0;
        }

        public int[] getNotes() {
            return notes;
        }

        public void setNotes(int[] notes) {
            this.notes = notes;
        }

        public int getCardinality() {
            return cardinality;
        }

        public void setCardinality(int cardinality) {
            this.cardinality = cardinality;
        }
    }

    public class FeatureSubsetComparator implements Comparator {

        public int compare(Object o1, Object o2) {
            FeatureSubset fs1 = (FeatureSubset) o1;
            FeatureSubset fs2 = (FeatureSubset) o2;

            /*if (tempMerit >= best_merit) {
         tempSize = countFeatures(tempGroup);
         if (tempMerit > best_merit ||
             (tempSize < sizeOfBest)) {
           best_merit = tempMerit;
           m_bestGroup = (BitSet)(tempGroup.clone());
           sizeOfBest = tempSize;
           if (m_verbose) {
             System.out.println("New best subset ("
                                +Utils.doubleToString(Math.
                                                      abs(best_merit),8,5)
                                +"): "+printSubset(m_bestGroup));
           }
         }
       }*/

            if (fs1.getMerit() >= fs2.getMerit()) {
                if ((fs1.getMerit() > fs2.getMerit()) || (fs1.cardinality() < fs2.cardinality())) return 1;
                if ((fs1.getMerit() == fs2.getMerit()) && (fs1.cardinality() == fs2.cardinality())) return 0;
                return -1;
            } else {
                return -1;
            }
            /*if (fs1.getMerit() == fs2.getMerit()) return 0;
            if (fs1.getMerit() < fs2.getMerit()) return -1;*/

            //return 0;
        }
    }

    public class FeatureVoteComparator implements Comparator {

        public int compare(Object o1, Object o2) {
            FeatureVote fs1 = (FeatureVote) o1;
            FeatureVote fs2 = (FeatureVote) o2;

            /*if (tempMerit >= best_merit) {
         tempSize = countFeatures(tempGroup);
         if (tempMerit > best_merit ||
             (tempSize < sizeOfBest)) {
           best_merit = tempMerit;
           m_bestGroup = (BitSet)(tempGroup.clone());
           sizeOfBest = tempSize;
           if (m_verbose) {
             System.out.println("New best subset ("
                                +Utils.doubleToString(Math.
                                                      abs(best_merit),8,5)
                                +"): "+printSubset(m_bestGroup));
           }
         }
       }*/
            if ((fs1.getMerit() * (fs1.cardinality() / m_numAttribs)) - (fs2.getMerit() * (fs2.cardinality() / m_numAttribs)) > 0) {
                return 1;
            } else {
                return -1;
            }

            /*if (fs1.getMerit() >= fs2.getMerit()) {
                if ((fs1.getMerit() > fs2.getMerit()) || (fs1.cardinality() < fs2.cardinality())) return 1;
            } else {
                return -1;
            }

            return 0;*/
        }
    }

}