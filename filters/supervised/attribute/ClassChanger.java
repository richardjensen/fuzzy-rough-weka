package weka.filters.supervised.attribute;

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
public class ClassChanger 
extends Filter
implements UnsupervisedFilter, OptionHandler, Serializable {

	/** for serialization */
	static final long serialVersionUID = 4752820393279263361L;

	/** Stores which values of nominal attribute are to be used for filtering.*/
	protected Range m_Values;

	/** Stores which value of a numeric attribute is to be used for filtering.*/
	protected double m_Value = 0;

	/** True if missing values should count as a match */
	protected boolean m_MatchMissingValues = false;

	/** Modify header for nominal attributes? */
	protected boolean m_ModifyHeader = false;

	/** If m_ModifyHeader, stores a mapping from old to new indexes */
	protected int [] m_NominalMapping;


	/** Proportion (in [0,1]) of attribute values to corrupt. 1 = all values, 0 = no values**/
	public String m_value1="0"; 

	/** The random number generator seed */
	public String m_value2 = "1";
	
	/**
	 * Returns a string describing this classifier
	 * @return a description of the classifier suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Replaces specified class value with different class value.";
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
			setValue1(knnString);
		} else {
			setValue1("0");
		}
		
		String seedString = Utils.getOption('S', options);
		if (seedString.length() != 0) {
			setValue2(seedString);
		} else {
			setValue2("1");
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
	public String getValue2() {

		return m_value2;
	}

	/**
	 * Sets the random number seed.
	 *
	 * @param newSeed the new random number seed.
	 */
	public void setValue2(String newSeed) {
		m_value2 = newSeed;
	}
	
	public String getValue1() {
		return m_value1;
	}

	public void setValue1(String t) {
		m_value1 = t;
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
		result.add(String.valueOf(getValue1()));
		
		result.add("-S");
		result.add(String.valueOf(getValue2()));
		
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

	private double max1;
	private double min1;
	private double stddev1;
	
	
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

			int m_numInstances = m_trainInstances.numInstances();

			//if the data has no decision feature, m_classIndex is negative
			int m_classIndex = m_trainInstances.classIndex();
			boolean nominal;
			int val1index=0, val2index=0;
			double val1=0,val2=0;
			
			if (m_trainInstances.classAttribute().isNominal()){
				nominal = true;
				val1index = m_trainInstances.classAttribute().indexOfValue(m_value1);
			}
			else {
				nominal=false;
				val1 = Double.valueOf(m_value1);
				val2 = Double.valueOf(m_value2);
			}

			setOutputFormat(m_trainInstances);
			
			
			
			for (int i=0;i<m_numInstances;i++) {
				if (nominal) {
					int value1 = (int) m_trainInstances.instance(i).value(m_classIndex);
					
					if (value1==val1index) {
						m_trainInstances.instance(i).setClassValue(m_value2);
					}

				}
				else {
					double value1 =  m_trainInstances.instance(i).value(m_classIndex);
					
					if (value1==val1) {
						m_trainInstances.instance(i).setClassValue(val2);
					}
				}
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
			
			if (values == 2) {
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
			double currValue = instance.value(a);
			double distort = r.nextGaussian();
			
			double newValue = currValue + (stddev1*distort);
			
			if (newValue>max1) newValue=max1;
			else if (newValue<min1) newValue=min1;
			
			instance.setValue(a, newValue); 
			
			//System.err.println(distort+" "+stddev[a]);
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
		runFilter(new ClassChanger(), argv);
	}
}
