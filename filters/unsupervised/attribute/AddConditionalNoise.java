package weka.filters.unsupervised.attribute;

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
 *    Copyright (C) 2009 Richard Jensen
 *
 */



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
import weka.filters.UnsupervisedFilter;


import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;
import java.util.Random;

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
public class AddConditionalNoise 
extends Filter
implements UnsupervisedFilter, OptionHandler, Serializable {

	/** for serialization */
	static final long serialVersionUID = 4752870393279263361L;

	/** Stores which values of nominal attribute are to be used for filtering.*/
	protected Range m_Values;

	/** Stores which value of a numeric attribute is to be used for filtering.*/
	protected double m_Value = 0;

	/** True if missing values should be used instead */
	protected boolean m_UseMissing = false;

	/** True if missing values should count as a match */
	protected boolean m_MatchMissingValues = false;

	/** Modify header for nominal attributes? */
	protected boolean m_ModifyHeader = false;

	/** If m_ModifyHeader, stores a mapping from old to new indexes */
	protected int [] m_NominalMapping;


	/** Proportion (in [0,1]) of attribute values to corrupt. 1 = all values, 0 = no values **/
	public double m_noiseProportion=0.01; 

	/** Amount of corruption to apply **/
	public double m_severity=1; 
	
	/** The random number generator seed */
	public int m_RandomSeed = 1;

	/**
	 * Returns a string describing this classifier
	 * @return a description of the classifier suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Adds noise to the conditional features (supports both nominal and numeric values).  For numeric values, the noise is Gaussian.";
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

		String knnString = Utils.getOption('K', options);
		if (knnString.length() != 0) {
			setThreshold(Double.valueOf(knnString));
		} else {
			setThreshold(0.2);
		}
		
		String sev = Utils.getOption('S', options);
		if (sev.length() != 0) {
			setSeverity(Double.valueOf(sev));
		} else {
			setSeverity(1);
		}

		if (Utils.getFlag('M', options)) {
			setUseMissing(true);
		}

		String seedString = Utils.getOption('S', options);
		if (seedString.length() != 0) {
			setRandomSeed(Integer.parseInt(seedString));
		} else {
			setRandomSeed(1);
		}
	}


	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String randomSeedTipText() {

		return "Random number seed.";
	}

	/**
	 * Gets the random number seed.
	 *
	 * @return the random number seed.
	 */
	public int getRandomSeed() {

		return m_RandomSeed;
	}

	/**
	 * Sets the random number seed.
	 *
	 * @param newSeed the new random number seed.
	 */
	public void setRandomSeed(int newSeed) {
		m_RandomSeed = newSeed;
	}


	/**
	 * Gets the flag if missing values are to be used.
	 *
	 * @return the flag missing values.
	 */
	public boolean getUseMissing() {

		return m_UseMissing;
	}

	/**
	 * Sets the flag if missing values are to be used.
	 *
	 * @param newUseMissing the new flag value.
	 */
	public void setUseMissing(boolean newUseMissing) {
		m_UseMissing = newUseMissing;
	}

	public double getThreshold() {
		return m_noiseProportion;
	}

	public void setThreshold(double t) {
		if (t>1 || t<0) t=0.2;

		m_noiseProportion = t;
	}

	public double getSeverity() {
		return m_severity;
	}

	public void setSeverity(double s) {
		m_severity = s;
	}

	/**
	 * Gets the current settings of FuzzyRoughSubsetEval
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions ()  {
		Vector<String>	result;

		result = new Vector<String>();

		result.add("-K");
		result.add(String.valueOf(getThreshold()));

		result.add("-S");
		result.add(String.valueOf(getSeverity()));

		
		if (getUseMissing()) {
			result.add("-M");
		}

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

	private double[] max=null;
	private double[] min=null;
	private double[] stddev=null;


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

			Instances m_trainInstances = getInputFormat();
			m_trainInstances.deleteWithMissingClass();

			int m_numAttribs = m_trainInstances.numAttributes();
			int m_numInstances = m_trainInstances.numInstances();

			//if the data has no decision feature, m_classIndex is negative
			int m_classIndex = m_trainInstances.classIndex();


			max = new double[m_numAttribs];
			min = new double[m_numAttribs];
			stddev = new double[m_numAttribs];

			for (int a=0;a<m_numAttribs;a++) {

				if (a!=m_classIndex && m_trainInstances.attribute(a).isNumeric()) {
					stddev[a] = (m_trainInstances.attributeStats(a)).numericStats.stdDev;
					max[a] = (m_trainInstances.attributeStats(a)).numericStats.max;
					min[a] = (m_trainInstances.attributeStats(a)).numericStats.min;
				}

			}

			//supervised
			if (m_classIndex>=0) {	
				m_numAttribs--;
			}

			setOutputFormat(m_trainInstances);

			Random random = new Random(m_RandomSeed);

			int instIndex;
			int attrIndex;
			int iterations = (int)(m_noiseProportion*m_numInstances*m_numAttribs);

			System.err.println("Noisy values (conditional): "+iterations);

			for (int i=0;i<iterations;i++) {
				instIndex = (int)(random.nextInt(m_numInstances));
				attrIndex = (int)(random.nextInt(m_numAttribs));

				while (attrIndex==m_classIndex) attrIndex = (int)(random.nextInt(m_numAttribs));

				corrupt(m_trainInstances,instIndex,attrIndex,random);
			}

			for (int i = 0; i < getInputFormat().numInstances(); i++) {
				push(getInputFormat().instance(i));
			}

		}
		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;
		return (numPendingOutput() != 0);
	}

	public void corrupt(Instances instances, int i, int a, Random r) {
		Instance instance = instances.instance(i);

		if (instance.attribute(a).isNominal()) {
			int values = instance.attribute(a).numValues();
			int currValue = (int) instance.value(a);

			if (m_UseMissing) {instance.setMissing(a);}
			else if (values == 1) {

			}
			else if (values == 2) {
				instance.setValue(a, (double) ((currValue+1)% 2));
			}
			else {
				while (true) {
					int newValue = (int) (r.nextDouble() * (double) values);

					// have we found a new value?
					if (newValue != currValue) { 
						instance.setValue(a, (double) newValue); 
						break;
					}
				}
			}

		}
		else if (instance.attribute(a).isNumeric()) {
			if (m_UseMissing) {instance.setMissing(a);}
			else {
				double currValue = instance.value(a);
				double distort = r.nextGaussian();

				double newValue = currValue + (stddev[a]*distort*m_severity);

				if (newValue>max[a]) newValue=max[a];
				else if (newValue<min[a]) newValue=min[a];

				instance.setValue(a, newValue); 

				//System.err.println(distort+" "+stddev[a]);
			}
		}
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
		runFilter(new AddConditionalNoise(), argv);
	}
}
