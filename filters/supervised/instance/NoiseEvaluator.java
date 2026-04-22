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



import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.AttributeTransformer;
import weka.attributeSelection.FuzzyRoughSubsetEval;
import weka.attributeSelection.HillClimber;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.UnsupervisedAttributeEvaluator;
import weka.attributeSelection.UnsupervisedSubsetEvaluator;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionUtils;
import weka.core.SparseInstance;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.filters.supervised.attribute.AddClassNoise;
import weka.filters.unsupervised.attribute.AddConditionalNoise;

import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.measure.*;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity3;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormLukasiewicz;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
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
public class NoiseEvaluator 
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

	private Filter m_FilterOne = new AddClassNoise();

	private Filter m_FilterTwo = new AddConditionalNoise();

	/** the attribute selection evaluation object */
	private weka.attributeSelection.AttributeSelection m_trainSelector = new weka.attributeSelection.AttributeSelection();

	/** the attribute evaluator to use */
	private ASEvaluation m_ASEvaluator = new FuzzyRoughSubsetEval();

	/** the search method if any */
	private ASSearch m_ASSearch = new HillClimber();

	/** holds the selected attributes  */
	private int [] m_SelectedAttributes;

	public double m_noiseProportion=0.01; 
	
	public Similarity m_Similarity = new Similarity3();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity = new SimilarityEq();
	public FuzzyMeasure m_Measure= new WeakGamma();

	public TNorm m_TNorm = new TNormLukasiewicz();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();
	
	/**
	 * Returns a string describing this classifier
	 * @return a description of the classifier suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Applies two filters to data, reduces it via the chosen evaluator and then reduces the non-noisy version of the data. Filter1 is applied first, then filter2.";
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

		String optionString = Utils.getOption('E',options);
		if (optionString.length() != 0) {
			optionString = optionString.trim();
			// split a quoted evaluator name from its options (if any)
			int breakLoc = optionString.indexOf(' ');
			String evalClassName = optionString;
			String evalOptionsString = "";
			String [] evalOptions=null;
			if (breakLoc != -1) {
				evalClassName = optionString.substring(0, breakLoc);
				evalOptionsString = optionString.substring(breakLoc).trim();
				evalOptions = Utils.splitOptions(evalOptionsString);
			}
			setEvaluator(ASEvaluation.forName(evalClassName, evalOptions));
		}

		if (m_ASEvaluator instanceof AttributeEvaluator) {
			setSearch(new Ranker());
		}

		optionString = Utils.getOption('S',options);
		if (optionString.length() != 0) {
			optionString = optionString.trim();
			int breakLoc = optionString.indexOf(' ');
			String SearchClassName = optionString;
			String SearchOptionsString = "";
			String [] SearchOptions=null;
			if (breakLoc != -1) {
				SearchClassName = optionString.substring(0, breakLoc);
				SearchOptionsString = optionString.substring(breakLoc).trim();
				SearchOptions = Utils.splitOptions(SearchOptionsString);
			}
			setSearch(ASSearch.forName(SearchClassName, SearchOptions));
		}

		optionString = Utils.getOption('N', options);
		if(optionString.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(optionString);
			if(nnSearchClassSpec.length == 0) { 
				throw new Exception("Invalid Filter1 specification string."); 
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setFilter1( (Filter) Utils.forName( Filter.class, className, nnSearchClassSpec) );
		}
		else {
			setFilter1(new AddClassNoise());
		}

		optionString = Utils.getOption('M', options);
		if(optionString.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(optionString);
			if(nnSearchClassSpec.length == 0) { 
				throw new Exception("Invalid Filter2 specification string."); 
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setFilter2( (Filter) Utils.forName( Filter.class, className, nnSearchClassSpec) );
		}
		else {
			setFilter2(new AddConditionalNoise());
		}

		String knnString = Utils.getOption('K', options);
		if (knnString.length() != 0) {
			setThreshold(Double.valueOf(knnString));
		} else {
			setThreshold(0.2);
		}
	}


	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String evaluatorTipText() {

		return "Determines how attributes/attribute subsets are evaluated.";
	}

	public void setFilter1(Filter cn) {
		m_FilterOne = cn;
	}

	public Filter getFilter1() {
		return m_FilterOne;
	}


	public void setFilter2(Filter cn) {
		m_FilterTwo = cn;
	}

	public Filter getFilter2() {
		return m_FilterTwo;
	}

	public double getThreshold() {
		return m_noiseProportion;
	}

	public void setThreshold(double t) {
		m_noiseProportion = t;
	}
	
	/**
	 * set attribute/subset evaluator
	 * 
	 * @param evaluator the evaluator to use
	 */
	public void setEvaluator(ASEvaluation evaluator) {
		m_ASEvaluator = evaluator;
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String searchTipText() {

		return "Determines the search method.";
	}

	/**
	 * Set search class
	 * 
	 * @param search the search class to use
	 */
	public void setSearch(ASSearch search) {
		m_ASSearch = search;
	}

	/**
	 * Get the name of the attribute/subset evaluator
	 *
	 * @return the name of the attribute/subset evaluator as a string
	 */
	public ASEvaluation getEvaluator() {

		return m_ASEvaluator;
	}

	/**
	 * Get the name of the search method
	 *
	 * @return the name of the search method as a string
	 */
	public ASSearch getSearch() {

		return m_ASSearch;
	}


	/**
	 * Gets the current settings of FuzzyRoughSubsetEval
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions ()  {
		Vector<String>	result;

		result = new Vector<String>();




		result.add("-E");
		result.add(getEvaluator().getClass().getName());

		result.add("-S");
		result.add(getSearch().getClass().getName());

		result.add("-N");
		result.add(getFilter1().getClass().getName());

		result.add("-M");
		result.add(getFilter2().getClass().getName());

		result.add("-K");
		result.add(String.valueOf(getThreshold()));
		
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
	public boolean input(Instance instance) throws Exception {

		if (getInputFormat() == null) {
			throw new IllegalStateException("No input instance format defined");
		}
		if (m_NewBatch) {
			resetQueue();
			m_NewBatch = false;
		}
		if (isFirstBatchDone()) {
			convertInstance(instance);
			return true;
		} 

		bufferInput(instance);
		
		return false;
		
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

			m_FilterOne.setInputFormat(m_trainInstances);
			m_FilterTwo.setInputFormat(m_trainInstances);
			
			//apply the two filters
			Instances newData = Filter.useFilter(Filter.useFilter(m_trainInstances,m_FilterOne), m_FilterTwo);
			
			//set up and apply feature selection
			m_trainSelector.setEvaluator(m_ASEvaluator);
			m_trainSelector.setSearch(m_ASSearch);
			
			m_trainSelector.SelectAttributes(newData);
			m_SelectedAttributes = m_trainSelector.selectedAttributes();

			if (m_SelectedAttributes == null) {
				throw new Exception("No selected attributes\n");
			}

			setOutputFormat();
			
			int m_numAttribs = m_trainInstances.numAttributes();
			int m_numInstances = m_trainInstances.numInstances();
			//System.err.println(m_numAttribs + " "+m_numInstances);
			
			//if the data has no decision feature, m_classIndex is negative
			int m_classIndex = m_trainInstances.classIndex();

			//supervised
			if (m_classIndex>=0) {
				if (m_trainInstances.attribute(m_classIndex).isNumeric()) {
					System.err.println("Numeric class");
				
					m_DecisionSimilarity = m_Similarity;
				}
				else m_DecisionSimilarity = m_SimilarityEq;

			}

			m_Similarity.setInstances(m_trainInstances);
			m_DecisionSimilarity.setInstances(m_trainInstances);
			m_SimilarityEq.setInstances(m_trainInstances);
			m_composition = m_Similarity.getTNorm();

			m_Measure.set(m_Similarity,m_DecisionSimilarity,m_TNorm,m_composition,m_Implicator,m_SNorm,m_numInstances,m_numAttribs,m_classIndex,m_trainInstances);
			
			BitSet subset = new BitSet(m_numAttribs);
			
			for (int a=0;a<m_SelectedAttributes.length;a++) {
				subset.set(m_SelectedAttributes[a]);
			}

			double denom = m_Measure.getConsistency();
			double gamma = m_Measure.calculate(subset);
			
			System.err.println("Dataset consistency: "+denom);
			System.err.println("WeakGamma for subset: "+gamma/denom);

			
			// Convert pending input instances
			for (int i = 0; i < getInputFormat().numInstances(); i++) {
				convertInstance(getInputFormat().instance(i));
			}
			
			flushInput();
			
		}

		m_NewBatch = true;
		m_FirstBatchDone = true;
		return (numPendingOutput() != 0);
	}
	/**
	 * Set the output format. Takes the currently defined attribute set 
	 * m_InputFormat and calls setOutputFormat(Instances) appropriately.
	 * 
	 * @throws Exception if something goes wrong
	 */
	protected void setOutputFormat() throws Exception {
		Instances informat;

		if (m_SelectedAttributes == null) {
			setOutputFormat(null);
			return;
		}

		ArrayList<Attribute> attributes = new ArrayList<Attribute>(m_SelectedAttributes.length);

		int i;
		if (m_ASEvaluator instanceof AttributeTransformer) {
			informat = ((AttributeTransformer)m_ASEvaluator).transformedHeader();
		} else {
			informat = getInputFormat();
		}

		for (i=0;i < m_SelectedAttributes.length;i++) {
			attributes.add((Attribute)informat.attribute(m_SelectedAttributes[i]).copy());
		}

		Instances outputFormat = 
			new Instances(getInputFormat().relationName(), attributes, 0);


		if (!(m_ASEvaluator instanceof UnsupervisedSubsetEvaluator) &&
				!(m_ASEvaluator instanceof UnsupervisedAttributeEvaluator)) {
			outputFormat.setClassIndex(m_SelectedAttributes.length - 1);
		}

		setOutputFormat(outputFormat);  
	}

	/**
	 * Convert a single instance over. Selected attributes only are transfered.
	 * The converted instance is added to the end of
	 * the output queue.
	 *
	 * @param instance the instance to convert
	 * @throws Exception if something goes wrong
	 */
	protected void convertInstance(Instance instance) throws Exception {
		double[] newVals = new double[getOutputFormat().numAttributes()];

		if (m_ASEvaluator instanceof AttributeTransformer) {
			Instance tempInstance = ((AttributeTransformer)m_ASEvaluator).
			convertInstance(instance);
			for (int i = 0; i < m_SelectedAttributes.length; i++) {
				int current = m_SelectedAttributes[i];
				newVals[i] = tempInstance.value(current);
			}
		} else {
			for (int i = 0; i < m_SelectedAttributes.length; i++) {
				int current = m_SelectedAttributes[i];
				newVals[i] = instance.value(current);
			}
		}
		if (instance instanceof SparseInstance) {
			push(new SparseInstance(instance.weight(), newVals));
		} else {
			push(new DenseInstance(instance.weight(), newVals));
		}
	}

	/**
	 * set options to their default values
	 */
	protected void resetOptions() {
		m_trainSelector = new weka.attributeSelection.AttributeSelection();
		setEvaluator(new FuzzyRoughSubsetEval());
		setSearch(new HillClimber());
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
		runFilter(new NoiseEvaluator(), argv);
	}
}
