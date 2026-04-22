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
import weka.filters.SupervisedFilter;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.measure.*;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity2;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormLukasiewicz;

import java.io.Serializable;
import java.util.Enumeration;
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
public class BoundaryRegionFilter 
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

	/** If m_ModifyHeader, stores a mapping from old to new indexes */
	protected int [] m_NominalMapping;

	public Similarity m_Similarity = new Similarity2();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity = new SimilarityEq();
	public FuzzyMeasure m_Measure= new BoundaryRegion();

	/** Threshold for deciding the quality of objects to retain**/
	public double m_threshold=0; 

	public TNorm m_TNorm = new TNormLukasiewicz();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();

	public boolean iterative=false;

	/**
	 * Returns a string describing this classifier
	 * @return a description of the classifier suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Calculates the contribution to the measure for each object and removes objects with values less than threshold.";
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
	public void setOptions (String[] options) throws Exception {
		String optionString;	   

		optionString = Utils.getOption('Z', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid FuzzyMeasure specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setMeasure( (FuzzyMeasure) Utils.forName( FuzzyMeasure.class, className, moreOptions) );
		}
		else {
			setMeasure(new BoundaryRegion());
		}



		optionString = Utils.getOption('R', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid Similarity specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setSimilarity( (Similarity) Utils.forName( Similarity.class, className, moreOptions) );
		}
		else {
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



	public Similarity getSimilarity() {
		return m_Similarity;
	}

	public void setSimilarity(Similarity s) {
		m_Similarity = s;
	}

	public double getThreshold() {
		return m_threshold;
	}

	public void setThreshold(double t) {
		m_threshold = t;
	}

	public boolean getIterative() {
		return iterative;
	}

	public void setIterative(boolean b) {
		iterative=b;
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
		result.add((m_Measure.getClass().getName() + " " +
				Utils.joinOptions(m_Measure.getOptions())).trim());


		result.add("-R");
		result.add((m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim());

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
				Instances m_trainInstances = getInputFormat();


				m_trainInstances.deleteWithMissingClass();

				int m_numAttribs = m_trainInstances.numAttributes();
				int m_numInstances = m_trainInstances.numInstances();

				System.err.println(m_numAttribs + " "+m_numInstances);

				//if the data has no decision feature, m_classIndex is negative
				int m_classIndex = m_trainInstances.classIndex();

				//supervised
				if (m_classIndex>=0) {
					boolean m_isNumeric = m_trainInstances.attribute(m_classIndex).isNumeric();

					if (m_isNumeric) {
						m_DecisionSimilarity = m_Similarity;
					}
					else m_DecisionSimilarity = m_SimilarityEq;

				}

				m_Similarity.setInstances(m_trainInstances);
				m_DecisionSimilarity.setInstances(m_trainInstances);
				m_SimilarityEq.setInstances(m_trainInstances);
				m_composition = m_Similarity.getTNorm();

				m_Measure.set(m_Similarity,m_DecisionSimilarity,m_TNorm,m_composition,m_Implicator,m_SNorm,m_numInstances,m_numAttribs,m_classIndex,m_trainInstances);

				double[] mems = m_Measure.objectMemberships();

				setOutputFormat(m_trainInstances);

				for (int i = 0; i< mems.length; i++) {
					//only report those instances whose membership to the boundary region is > 0
					if (mems[i]>m_threshold) System.err.println("Instance "+i+": "+mems[i]);
					push(getInputFormat().instance(i));
				}

			
	
			
		}
		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;
		return (numPendingOutput() != 0);
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
		runFilter(new BoundaryRegionFilter(), argv);
	}
}
