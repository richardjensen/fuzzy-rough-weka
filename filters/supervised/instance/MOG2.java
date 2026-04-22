package weka.filters.supervised.instance;

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Copyright (C) 2015 Richard Jensen
 *
 */



import weka.classifiers.Classifier;
import weka.classifiers.fuzzy.VQNN;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionUtils;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * Filters instances according to the value of an attribute.
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 <!-- options-end -->
 *
 * @author Richard Jensen
 * @version $Revision: 1.12 $
 */
public class MOG2 
extends Filter
implements SupervisedFilter, OptionHandler, Serializable {

	/** for serialization */
	static final long serialVersionUID = 4752870393679263361L;

	/** Stores which values of nominal attribute are to be used for filtering.*/
	protected Range m_Values;

	/** Stores which value of a numeric attribute is to be used for filtering.*/
	protected double m_Value = 0;

	/** True if missing values should count as a match */
	protected boolean m_MatchMissingValues = false;

	/** Modify header for nominal attributes? */
	protected boolean m_ModifyHeader = false;
	long seed = 4;

	/** If m_ModifyHeader, stores a mapping from old to new indexes */
	protected int [] m_NominalMapping;

	public Classifier m_classifier = new VQNN();

	public boolean version2=false;

	/** Threshold for deciding the quality of objects to retain**/
	public double m_threshold=0.8; 


	/** Proportion of objects to retain compared to the number of objects in the majority class**/
	public double m_proportion=0.8;


	public boolean iterative=false;

	/**
	 * Returns a string describing this classifier
	 * @return a description of the classifier suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "(version 2) Balances the dataset by using a classifier to classify constrained randomly generated instances";
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
				"\tNumeric value to be used for selection on numeric\n"+
						"\tattribute.\n"+
						"\tInstances with values smaller than given value will\n"+
						"\tbe selected. (default 0)",
						"S", 1, "-S <num>"));
		newVector.addElement(new Option(
				"\tRange of label indices to be used for selection on\n"+
						"\tnominal attribute.\n"+
						"\tFirst and last are valid indexes. (default all values)",
						"L", 1, "-L <index1,index2-index4,...>"));
		newVector.addElement(new Option(
				"\tMissing values count as a match. This setting is\n"+
						"\tindependent of the -V option.\n"+
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
	 *
	   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * 
	   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 *
	 **/
	public void setOptions (String[] options)
			throws Exception {

		String optionString;	   

		optionString = Utils.getOption('Z', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid Classifier specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setClassifier( (Classifier) Utils.forName( Classifier.class, className, moreOptions) );
		}
		else {
			setClassifier(new VQNN());
		}


		String knnString = Utils.getOption('K', options);
		if (knnString.length() != 0) {
			setThreshold(Double.valueOf(knnString));
		} else {
			setThreshold(0.8);
		}

		knnString = Utils.getOption('P', options);
		if (knnString.length() != 0) {
			setProportion(Double.valueOf(knnString));
		} else {
			setProportion(0.8);
		}

	}

	public void setClassifier(Classifier fe) {
		m_classifier = fe;
	}

	public Classifier getClassifier() {
		return m_classifier;
	}


	public double getThreshold() {
		return m_threshold;
	}

	public void setThreshold(double t) {
		m_threshold = t;
	}

	public double getProportion() {
		return m_proportion;
	}

	public void setProportion(double p) {
		m_proportion = p;
	}


	/**
	 * Gets the current settings of FuzzyRoughSubsetEval
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions ()  {
		Vector<String>	result;

		result = new Vector<String>();

		result.add("-Z");
		result.add((m_classifier.getClass().getName() + " " +
				Utils.joinOptions(((OptionHandler)m_classifier).getOptions())).trim());

		result.add("-K");
		result.add(String.valueOf(getThreshold()));

		result.add("-P");
		result.add(String.valueOf(getProportion()));

		return result.toArray(new String[result.size()]);
	}

	/** 
	 * Returns the Capabilities of this filter.
	 *
	 * @return            the capabilities of this object
	 * @see               Capabilities
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
	 * @param instanceInfo an Instances object containing the input instance
	 * structure (any instances contained in the object are ignored - only the
	 * structure is required).
	 * @throws UnsupportedAttributeTypeException if the specified attribute
	 * is neither numeric or nominal.
	 * @return true because outputFormat can be collected immediately
	 */
	public boolean setInputFormat(Instances instanceInfo) throws Exception {
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
	 * collected with output().
	 * @throws IllegalStateException if no input structure has been defined
	 */
	public boolean input(Instance instance) {

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

	/**
	 * Signify that this batch of input to the filter is finished. If the filter
	 * requires all instances prior to filtering, output() may now be called
	 * to retrieve the filtered instances.
	 *
	 * @return true if there are instances pending output.
	 * @throws IllegalStateException if no input structure has been defined.
	 * @throws Exception if there is a problem during the attribute selection.
	 */
	public boolean batchFinished() throws Exception {

		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}

		if (!isFirstBatchDone()) {

			//sort the instances according to the class attribute
			getInputFormat().sort(getInputFormat().classIndex());
			Instances m_trainInstances = getInputFormat();

			int numberOfClasses=getInputFormat().numClasses();

			// Create an index of where each class value starts
			int[] classIndices = new int[getInputFormat().numClasses() + 1];
			int[] classCount = new int[getInputFormat().numClasses() + 1];
			double[][] max = new double[getInputFormat().numClasses()][getInputFormat().numAttributes()];
			double[][] min = new double[getInputFormat().numClasses()][getInputFormat().numAttributes()];

			for (int c=0;c<numberOfClasses;c++) {
				for (int a = 0;a<getInputFormat().numAttributes();a++) {
					max[c][a] = Double.NEGATIVE_INFINITY;
					min[c][a] = Double.POSITIVE_INFINITY;
				}
			}

			int currentClass = 0;
			classIndices[currentClass] = 0;

			for (int i = 0; i < getInputFormat().numInstances(); i++) {
				Instance current = getInputFormat().instance(i);

				if (current.classValue() != currentClass) {
					for (int j = currentClass + 1; j <= current.classValue(); j++) {
						classIndices[j] = i;
					}          
					currentClass = (int) current.classValue();
				}

				//update max and min values
				for (int a = 0;a<getInputFormat().numAttributes();a++) {
					if (current.value(a)<min[currentClass][a]) min[currentClass][a]=current.value(a);
					if (current.value(a)>max[currentClass][a]) max[currentClass][a]=current.value(a);
				}

				classCount[currentClass]++;
			}

			if (currentClass <= getInputFormat().numClasses()) {
				for (int j = currentClass + 1; j < classIndices.length; j++) {
					classIndices[j] = getInputFormat().numInstances();
				}
			}

			//find the majority class
			int majorityClass=-1;
			int majorityClassCount=0;
			int totalToCreate=0;

			for (int c=0;c<numberOfClasses;c++) {
				if (classCount[c]>majorityClassCount) {
					majorityClassCount = classCount[c];
					majorityClass = c;
				}
			}


			//build the classifier to be used for determining classes of new objects
			m_classifier.buildClassifier(m_trainInstances);
			setOutputFormat(m_trainInstances);
			//contains the new Instance objects that will be added to the original dataset eventually
			ArrayList<Instance> toAdd = new ArrayList<Instance>();
			Random random = new Random(seed);

			//for the other classes, calculate how many objects need to be created to balance the dataset
			int[] toCreate = new int[numberOfClasses];

			for (int c=0;c<numberOfClasses;c++) {
				if (c!=majorityClass) {
					toCreate[c] = (int) (m_proportion*(majorityClassCount - classCount[c]));
					totalToCreate+=toCreate[c];
				}
			}

			for (int c=0;c<numberOfClasses;c++) {
				if (c!=majorityClass) {
					for (int i=0;i<toCreate[c];i++) {
						//generate a random instance within the range of values seen for this class c
						Instance newInstance = generateRandomInstance(m_trainInstances,random,c,max,min);
						
						//classify the instance, find the class
						int cl = (int) m_classifier.classifyInstance(newInstance);

						//if this matches the current class, then add this instance
						if (c==cl) {
							newInstance.setClassValue(c);
							toAdd.add(newInstance);
						}
						else i--; //otherwise, try again
					}					

				}
			}


			//loop over toAdd list and add these instances
			for (Instance temp: toAdd) {
				push(temp);
			}

			for (int i = 0; i < getInputFormat().numInstances(); i++) {
				Instance temp = getInputFormat().instance(i);
				push(temp);
			}

			System.err.println("Added "+toAdd.size()+" instances");

		}
		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;
		return (numPendingOutput() != 0);
	}


	/***
	 * Generates a random instance within the scope of the set of instances belong to the specified class
	 * 
	 * @param m_train The training instances
	 * @param random Random number generator
	 * @param classIndex The class for which to generate an instance
	 * @param max 
	 * @param min 
	 * @return Instance
	 */
	public Instance generateRandomInstance(Instances m_train, Random random,int classIndex, double[][] max, double[][] min) {
		Instance toReturn = (Instance) m_train.firstInstance().copy();

		for (int a=0;a<m_train.numAttributes();a++) {
			if (a!=m_train.classIndex()) {
				if (m_train.attribute(a).isNominal()) { //if nominal, choose a random value
					//get the number of values this attribute can take
					int values = m_train.attribute(a).numValues();
					toReturn.setValue(a, m_train.attribute(a).value(random.nextInt(values)));
				}
				else {
					double value = min[classIndex][a] + (Math.abs(random.nextDouble())*(max[classIndex][a]-min[classIndex][a]));
					toReturn.setValue(a, value);
				}
			}
		}

		return toReturn;
	}

	/**
	 * set options to their default values
	 */
	protected void resetOptions() {

	}

	/**
	 * Returns the revision string.
	 * 
	 * @return		the revision
	 */
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 5499 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain arguments to the filter: 
	 * use -h for help
	 */
	public static void main(String [] argv) {
		runFilter(new MOG2(), argv);
	}
}
