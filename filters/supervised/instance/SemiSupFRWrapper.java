package weka.filters.supervised.instance;


import weka.classifiers.fuzzy.*;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.fuzzy.implicator.*;
import weka.fuzzy.measure.FuzzyMeasure;
import weka.fuzzy.measure.WeakGamma;
import weka.fuzzy.similarity.*;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Enumeration;
import java.lang.Integer;
import java.util.Vector;

public class SemiSupFRWrapper
        extends Filter
        implements SupervisedFilter, OptionHandler {
    /**
     * for serialization
     */
    static final long serialVersionUID = 4752870393679263361L;

    /**
     * Stores which values of nominal attribute are to be used for
     * filtering.
     */
    protected Range m_Values;

    /**
     * Stores which value of a numeric attribute is to be used for
     * filtering.
     */
    protected double m_Value = 0;

    /**
     * True if missing values should count as a match
     */
    protected boolean m_MatchMissingValues = false;

    /**
     * Modify header for nominal attributes?
     */
    protected boolean m_ModifyHeader = false;

    /**
     * If m_ModifyHeader, stores a mapping from old to new indexes
     */
    protected int[] m_NominalMapping;

    /**
     * class type nominal/real-valued
     */
    protected int m_ClassType;

    /**
     * The number of class values (or 1 if predicting numeric).
     */
    protected int m_NumClasses;

    public Similarity m_Similarity = new Similarity4();
    public Similarity m_SimilarityEq = new SimilarityEq();
    public Similarity m_DecisionSimilarity = new SimilarityEq();
    public FuzzyMeasure m_Measure = new WeakGamma();

    /**
     * Threshold for deciding the quality of objects to retain*
     */
    public double m_threshold = 1;

    public TNorm m_TNorm = new TNormLukasiewicz();
    public TNorm m_composition = new TNormLukasiewicz();
    public Implicator m_Implicator = new ImplicatorLukasiewicz();
    public SNorm m_SNorm = new SNormLukasiewicz();

    public boolean iterative = false;

    /**
     * Returns a string describing this classifier
     *
     * @return a description of the classifier suitable for
     *         displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        return "Calculates the belonging of each unlabelled object to the lower approxes of the labelled classes.";
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration<Option> listOptions() {

        Vector<Option> newVector = new Vector<Option>(5);

        newVector.addElement(new Option(
                "\tChoose attribute to be used for selection.",
                "C", 1, "-C <num>"));
        newVector.addElement(new Option(
                "\tNumeric value to be used for selection on numeric\n" +
                        "\tattribute.\n" +
                        "\tInstances with values smaller than given value will\n" +
                        "\tbe selected. (default 0)",
                "S", 1, "-S <num>"));
        newVector.addElement(new Option(
                "\tRange of label indices to be used for selection on\n" +
                        "\tnominal attribute.\n" +
                        "\tFirst and last are valid indexes. (default all values)",
                "L", 1, "-L <index1,index2-index4,...>"));
        newVector.addElement(new Option(
                "\tMissing values count as a match. This setting is\n" +
                        "\tindependent of the -V option.\n" +
                        "\t(default missing values don't match)",
                "M", 0, "-M"));
        newVector.addElement(new Option(
                "\tInvert matching sense.",
                "V", 0, "-V"));
        newVector.addElement(new Option(
                "\tWhen selecting on nominal attributes, removes header\n"
                        + "\treferences to excluded values.",
                "H", 0, "-H"));

        return newVector.elements();
    }


    /**
     * Parses and sets a given list of options. <p/>
     * <p/>
     * <!-- options-start -->
     * Valid options are: <p/>
     * <p/>
     * <p/>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options)
            throws Exception {

        String optionString;

        optionString = Utils.getOption('Z', options);
        if (optionString.length() != 0) {
            String moreOptions[] =
                    Utils.splitOptions(optionString);
            if (moreOptions.length == 0) {
                throw new Exception("Invalid FuzzyMeasure specification string.");
            }
            String className = moreOptions[0];
            moreOptions[0] = "";

            setMeasure((FuzzyMeasure) Utils.forName(
                    FuzzyMeasure.class, className, moreOptions));
        } else {
            setMeasure(new WeakGamma());
        }


        optionString = Utils.getOption('I', options);
        if (optionString.length() != 0) {
            String moreOptions[] =
                    Utils.splitOptions(optionString);
            if (moreOptions.length == 0) {
                throw new Exception("Invalid Implicator specification string.");
            }
            String className = moreOptions[0];
            moreOptions[0] = "";

            setImplicator((Implicator) Utils.forName(
                    Implicator.class, className, moreOptions));
        } else {
            setImplicator(new ImplicatorLukasiewicz());
        }


        optionString = Utils.getOption('T', options);
        if (optionString.length() != 0) {
            String moreOptions[] =
                    Utils.splitOptions(optionString);
            if (moreOptions.length == 0) {
                throw new Exception("Invalid TNorm specification string.");
            }
            String className = moreOptions[0];
            moreOptions[0] = "";

            setTNorm((TNorm) Utils.forName(TNorm.class, className,
                    moreOptions));
        } else {
            setTNorm(new TNormLukasiewicz());
        }

        optionString = Utils.getOption('R', options);
        if (optionString.length() != 0) {
            String moreOptions[] =
                    Utils.splitOptions(optionString);
            if (moreOptions.length == 0) {
                throw new Exception("Invalid Similarity specification string.");
            }
            String className = moreOptions[0];
            moreOptions[0] = "";

            setSimilarity((Similarity) Utils.forName(
                    Similarity.class, className, moreOptions));
        } else {
            setSimilarity(new Similarity2());
        }

        String knnString = Utils.getOption('K', options);
        if (knnString.length() != 0) {
            setThreshold(Double.valueOf(knnString));
        } else {
            setThreshold(1);
        }

        setIterative(Utils.getFlag('B', options));
    }

    public void setMeasure(FuzzyMeasure fe) {
        m_Measure = fe;
    }

    public FuzzyMeasure getMeasure() {
        return m_Measure;
    }


    public void setImplicator(Implicator impl) {
        m_Implicator = impl;
    }

    public Implicator getImplicator() {
        return m_Implicator;
    }

    //set the relation composition operator = tnorm
    public void setTNorm(TNorm tnorm) {
        m_TNorm = tnorm;
        //m_composition = tnorm;
        m_SNorm = tnorm.getAssociatedSNorm();
    }

    public TNorm getTNorm() {
        return m_TNorm;
    }

    public Similarity getSimilarity() {
        return m_Similarity;
    }

    public void setSimilarity(Similarity s) {
        m_Similarity = s;
    }

    public double getThreshold() {
        return m_threshold;
    }

    // NOT REQUIRED - NEEDS TO BE REMOVED
    public void setThreshold(double t) {
        m_threshold = t;
    }

    public boolean getIterative() {
        return iterative;
    }

    public void setIterative(boolean b) {
        iterative = b;
    }


    // ****************** FuzzyRoughSubsetEval ***********
    /**
     * Gets the current settings of FuzzyRoughSubsetEval
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions() {
        Vector<String> result;

        result = new Vector<String>();

        result.add("-Z");
        result.add((m_Measure.getClass().getName() + " " +
                Utils.joinOptions(m_Measure.getOptions())).trim());

        result.add("-I");
        result.add((m_Implicator.getClass().getName() + " " +
                Utils.joinOptions(m_Implicator.getOptions())).trim());

        result.add("-T");
        result.add((m_TNorm.getClass().getName() + " " +
                Utils.joinOptions(m_TNorm.getOptions())).trim());

        result.add("-R");
        result.add((m_Similarity.getClass().getName() + " " +
                Utils.joinOptions(m_Similarity.getOptions())).trim());

        result.add("-K");
        result.add(String.valueOf(getThreshold()));

        if (getIterative()) {
            result.add("-B");
        }

        return result.toArray(new String[result.size()]);
    }

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // attributes
        result.enableAllAttributes();
        result.enable(Capability.MISSING_VALUES);

        // class
        result.enableAllClasses();
        result.enable(Capability.MISSING_CLASS_VALUES);
        result.enable(Capability.NO_CLASS);

        return result;
    }

    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo an Instances object containing the input
     *                     instance
     *                     structure (any instances contained in the object are ignored
     *                     - only the
     *                     structure is required).
     * @return true because outputFormat can be collected immediately
     * @throws UnsupportedAttributeTypeException
     *          if the specified
     *          attribute
     *          is neither numeric or nominal.
     */
    public boolean setInputFormat(Instances instanceInfo) throws
            Exception {
        super.setInputFormat(instanceInfo);
        setOutputFormat(instanceInfo);
        return true;
    }

    /**
     * Input an instance for filtering. Filter requires all
     * training instances be read before producing output.
     *
     * @param instance the input instance
     * @return true if the filtered instance may now be
     *         collected with output().
     * @throws IllegalStateException if no input structure has been
     *                               defined
     */
    public boolean input(Instance instance) {
        //System.out.println("Input");
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }
        if (isFirstBatchDone()) {
            push(instance);
            return true;
        } else {
            bufferInput(instance);
            return false;
        }
    }

    //scan for class labels
    //remove unlabelled objects
    //remove labelled objects
    private void preprocessing(Instances masterCopy, ArrayList<Integer> labelledIndices, ArrayList<Integer> unlabelledIndices, double[] labels) {
        int numInstances = masterCopy.numInstances();
        for (int i = 0; i < numInstances; i++) {
            if (masterCopy.instance(i).classIsMissing()) {
                labels[i] = -1d;
                unlabelledIndices.add(i);
            } else {
                labels[i] = masterCopy.instance(i).classValue();
                labelledIndices.add(i);
            }
        }
    }

    private ArrayList<Integer> scanForUnlabelledIndices(Instances instances) {
        ArrayList<Integer> unlabelledIndices = new ArrayList<Integer>();
        int numInstances = instances.numInstances();
        for (int i = 0; i < numInstances; i++) {
            if (instances.instance(i).classIsMissing()) {
                unlabelledIndices.add(i);
            }
        }
        return unlabelledIndices;
    }

    private String getIndicesString(ArrayList<Integer> indices) {
        String indicesString = "";
        for (int i = 0; i < indices.size(); i++) {
            indicesString = indicesString + ", " + indices.get(i);
        }
        return indicesString;
    }

    /**
     * Signify that this batch of input to the filter is finished.
     * If the filter
     * requires all instances prior to filtering, output() may now
     * be called
     * to retrieve the filtered instances.
     *
     * @return true if there are instances pending output.
     * @throws IllegalStateException if no input structure has been
     *                               defined.
     * @throws Exception             if there is a problem during the attribute
     *                               selection.
     */
    public boolean batchFinished() throws Exception {

        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        Instances masterCopy = getInputFormat();

        Instances labelledData = new Instances(getInputFormat());
        labelledData.deleteWithMissingClass();

        FuzzyRoughNN frc = new FuzzyRoughNN();
        //set connectives and similarities
        frc.setSimilarity(m_Similarity);           
        frc.setImplicator(m_Implicator);
        frc.setTNorm(m_TNorm);
    
        

        ArrayList<Integer> unlabelledIndices = scanForUnlabelledIndices(masterCopy);
        System.out.println("Unlabelled Indices" + getIndicesString(unlabelledIndices));
        int unlabelledIndicesCount = unlabelledIndices.size() + 1;
        ArrayList<Integer> labelledIndices;
        
        if (!isFirstBatchDone()) {

            System.out.println("\n<< Entering Recursive Labelling Stage >>\n");

            while ((unlabelledIndices.size() > 0) && (unlabelledIndices.size() < unlabelledIndicesCount)) {

                unlabelledIndicesCount = unlabelledIndices.size();                    
                System.out.println("Unlabelled " + unlabelledIndices.size() + " Instance Indices" + getIndicesString(unlabelledIndices));

                frc.buildClassifier(labelledData);
                double[] distribution;
                int currentIndex;
                labelledIndices = new ArrayList<Integer>();
                
                for (int i = 0; i < unlabelledIndices.size(); i++) {
                    
                    currentIndex = unlabelledIndices.get(i);
                    distribution = frc.distributionForInstance(masterCopy.instance(currentIndex));

                    for (int j = 0; j < distribution.length; j++) {
                        if ((distribution[j]) == 1) {
                            masterCopy.instance(currentIndex).setClassValue(j);
                            System.out.println("Instance with Index = " + currentIndex + " labelled with " + j);
                            labelledData.add(masterCopy.instance(currentIndex));
                            labelledIndices.add(currentIndex);
                        }
                    }
                }
                
                unlabelledIndices.removeAll(labelledIndices);
                //if (labelledIndices.isEmpty()) System.out.println("Nothing Labelled, Exiting Next Loop");
            }

            System.out.println("\n<< Entering Best Class Labelling Stage >>\n");

            if (!unlabelledIndices.isEmpty()) {

                System.out.println("Remaining Unlabelled " + unlabelledIndices.size() + " Instance Indices" + getIndicesString(unlabelledIndices));
                double[] distribution;
                int currentIndex;

                for (int i = 0; i < unlabelledIndices.size(); i++) {

                    currentIndex = unlabelledIndices.get(i);
                    distribution = frc.distributionForInstance(masterCopy.instance(currentIndex));
                    
                    double bestMembership = -1d;
                    int bestClass = -1;
                    
                    for (int j = 0; j < distribution.length; j++) {
                        if ((distribution[j]) > bestMembership) {
                            bestClass = j;
                            bestMembership = distribution[j];
                            }
                        //System.out.println("lower approx membership = " + bestMembership);
                    }
                    masterCopy.instance(currentIndex).setClassValue(bestClass);
                    System.out.println("Instance with Index = " + currentIndex + " labelled with " + bestClass);
                }
            }

            //Put all of the newly labelled instances in the dataset for output
            for (int i = 0; i < getInputFormat().numInstances(); i++) {
                push(getInputFormat().instance(i));
            }
        }

        flushInput();

        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }
    //below are commented old code   
    /*public boolean batchFinished() throws Exception {
        System.out.println("batchfinished reached...");
        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }

        Instances masterCopy = getInputFormat();

        Instances labelledData = new Instances(getInputFormat());
        labelledData.deleteWithMissingClass();

        FuzzyRoughClassifier frc = new FuzzyRoughNN();

        ArrayList<Integer> labelledIndices = new ArrayList<Integer>();
        ArrayList<Integer> unlabelledIndices = new ArrayList<Integer>();
        double[] labels = new double[masterCopy.numInstances()];

        preprocessing(masterCopy, labelledIndices, unlabelledIndices, labels);

        if (!isFirstBatchDone()) {
            //int count = 0;


            //place holder for labelled data
            *//*Instances labelledData = new Instances(getInputFormat());
            labelledData.delete();*//*
            //place holder for unlabelled data
            *//*Instances unLabelledData = new Instances(getInputFormat());
            unLabelledData.delete();*//*
            //place holder that is used in each iteration to train a classifier
            //Instances updatedTrainingSet = new Instances(getInputFormat());
            //updatedTrainingSet.delete();

            //Instances copyOfTrain = new Instances(getInputFormat());
            //ArrayList availObs = new ArrayList();
            //remove the unlabelled instances from both sets of training data
            //labelledData.deleteWithMissingClass();
            //updatedTrainingSet.deleteWithMissingClass();


            //remove all objects from the unlabelled set
            //unLabelledData.delete();

            //System.out.println("unlabelled data size after trimming is: " + unLabelledData.numInstances());
            //System.out.println("labelled data size after trimming is: " + labelledData.numInstances());

            //System.out.println("m_unlab/lab done");

            //System.out.println("m_train size = " + masterCopy.numInstances());
            //int trainSize = masterCopy.numInstances();

            //Get the data and save unlabelled instance indexes to an ArrayList
            *//*for (int di = 0; di < trainSize; di++) {

                if ((masterCopy.instance(di)).classIsMissing()) {
                    availObs.add(new Integer(di));
                    System.out.println(di + " added to availObs");
                    unLabelledData.add(masterCopy.instance(di));
                    System.out.println("populating unlab data " + di);

                }

            }*//*

            //labelledData.compactify();
            //unLabelledData.compactify();
            *//*System.out.println("unlabelled data size is: " +
                    unLabelledData.numInstances());
            System.out.println("labelled data size is: " +
                    labelledData.numInstances());

            System.out.println(" data segregation is done ");*//*

            // Get the number of classes
            //m_NumClasses = labelledData.numClasses();

            //find out what type of attribute the decision class is
            //m_ClassType = labelledData.classAttribute().type();

            //System.out.println("    got to here   ");
            // Do membership stuff here - nested 'for' loop to get the upper and lower approx values for
            // each class
            //boolean finished=false;
            //int numUnassignedObs = availObs.size();
            //int updatedTrainingSetSize = 0;

            int unlabelledIndicesCount = unlabelledIndices.size();
            //while there are objects to assign
            //while (updatedTrainingSetSize < updatedTrainingSet.numInstances()) {
            while ((unlabelledIndicesCount > unlabelledIndices.size()) && (unlabelledIndices.size() > 0)) {
                unlabelledIndicesCount = unlabelledIndices.size();

                //updatedTrainingSetSize = updatedTrainingSet.numInstances();
                //System.out.println("updatedTrainingSet size:  " + updatedTrainingSet.numInstances());
                //build a classifier so that we can use info to classify unlabelled objects
                double[] distribution;
                //frc = new FuzzyRoughNN();
                frc.buildClassifier(labelledData);

                //find which ones are members of a given class to extent 1.0
                //Iterator ulit = availObs.iterator();

                //Array for all objects which must be removed
                //int[] removeObjects = new int[unLabelledData.numInstances()];
                int cnt = 0;
                //while(ulit.hasNext())
                int currentIndex;
                for (int i = 0; i < unlabelledIndices.size(); i++) {

                    currentIndex = unlabelledIndices.get(i);
                    //System.out.println("availObs size " + availObs.size());

                    //Integer uInt = (Integer) availObs.get(i);
                    //System.out.println("    uInt = " + uInt);
                    //Get the distribution for the unlabelled object
                    //System.out.println("    got to here  5 ");
                    distribution = frc.distributionForInstance(masterCopy.instance(currentIndex));
                    //System.out.println("    got to here  6 ");
                    //iterate over the distrib and find if any of the unlabelled objs are equal to 1
                    //System.out.println("distrib");
                    for (int j = 0; j < distribution.length; j++) {
                        //System.out.println("distrib2");
                        // If class membership is equal to 1.0 then assign that class to the object
                        if ((distribution[j]) == 1) {

                            System.out.println(" found object with memb = 1.0");
                            //add that class to the object and then the object to the updateable set of instances
                            Instance newlyLabelledInstance = masterCopy.instance(currentIndex);
                            newlyLabelledInstance.setClassValue(j);
                            labelledData.add(newlyLabelledInstance);
                            unlabelledIndices.remove(currentIndex);
                            //removeObjects[cnt] = i;
                            //System.out.println("i = " + i);
                            //System.out.println("updated training set size is now: " + updatedTrainingSet.numInstances());

                            //cnt++;
                            //System.out.println("cnt = " + cnt);
                            //numUnassignedObs--;
                            //break; //break out of the for loop
                            //} else if ((distribution[j]) < 1) {
                        } else {
                            System.out.println("this object is not equal to 1");
                        }
                    }

                }
                //Garbage collection - remove the objects that have been added to the updated training set
                *//*if (removeObjects.length < 1) System.out.println("no objects to remove");
                for (int cu = 0; cu < removeObjects.length; cu++) {
                    availObs.remove(removeObjects[cu]);
                    System.out.println("removed the object which has membership of 1");

                }*//*

            }

            // so if all else fails or if there are some objects that cannot be labelled by
            // self-learning then label the objects naievly
            double[] distribution;
            if (!unlabelledIndices.isEmpty()) {

                //System.out.println("    got to naive section");
                //Iterator unassign = availObs.iterator();
                //int cnt1 = 0;
                int currentIndex;
                for (int i = 0; i < unlabelledIndices.size(); i++) {

                    currentIndex = unlabelledIndices.get(i);
                    //while (unassign.hasNext()) {
                    //cnt1++;
                    //System.out.println("    got to naieve section while loop");
                    //Integer uInt = (Integer) (unassign.next());
                    //Get the distribution for the unlabelled object

                    distribution = frc.distributionForInstance(masterCopy.instance(currentIndex));
                    double bestMemb = -1d;
                    int bestClass = -1;
                    //Iterate over the distribution and get likely class membership
                    for (int j = 0; j < distribution.length; j++) {
                        if ((distribution[j]) > bestMemb) {

                            bestClass = j;
                            bestMemb = distribution[j];
                            //System.out.println("bestMemb is: " + bestMemb);
                        }
                        //System.out.println("Here is the distrib for object " + uInt + " and class " + j + " " + distribution[j]);

                    }

                    masterCopy.instance(currentIndex).setClassValue(bestClass);
                    //Instance newlyLabelledInstance = masterCopy.instance(currentIndex);
                    //System.out.println("bestClass: " + bestClass);
                    //newlyLabelledInstance.setClassValue(bestClass);
                    // add to the certain objects
                    //updatedTrainingSet.add(newlyLabelledInstance);
                }
            }


            //}


            //Put all of the newly labelled instances in the dataset for output
            for (int i = 0; i < getInputFormat().numInstances(); i++) {
                push(getInputFormat().instance(i));
            }


        }

        flushInput();

        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }*/

    /**
     * set options to their default values
     */
    protected void resetOptions() {

    }

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    //public String getRevision() {
    //return RevisionUtils.extract("$Revision: 5499 $");
    //}

    /**
     * Main method for testing this class.
     *
     * @param argv should contain arguments to the filter:
     *             use -h for help
     */
    public static void main(String[] argv) {
        //runFilter(new InstanceSelection(), argv);
    }
}