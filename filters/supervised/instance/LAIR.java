package weka.filters.supervised.instance;


import weka.core.AdditionalMeasureProducer;

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
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.SupervisedFilter;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity1;
import weka.fuzzy.similarity.Similarity3;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormKD;
import weka.fuzzy.tnorm.TNormLukasiewicz;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Enumeration;
import java.util.PriorityQueue;
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
public class LAIR 
extends Filter
implements SupervisedFilter, OptionHandler, Serializable, AdditionalMeasureProducer {

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

	public Similarity m_Similarity = new Similarity1();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity = new SimilarityEq();

	/** Threshold for deciding the quality of objects to retain
	 *  
	 *  For Similarity1, a threshold in the range 0.8-0.9 works well (i.e. good reductions whilst retaining accuracy)
	 *  For Similarity2, 0.5-0.7
	 *  For Similarity3, 0.0-0.3
	 ***/
	public double m_threshold=0.85; //generally a good setting for Similarity1 

	public TNorm m_TNorm = new TNormLukasiewicz();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();


	/** the various link types */
	final static int LOWER = 0;
	final static int VQ = 1;
	final static int LOWER_OWA = 2;

	public static final Tag[] TAGS_TYPE = {
			new Tag(LOWER, "WeakGamma"),
			new Tag(VQ, "VQRS"),
			new Tag(LOWER_OWA, "WeakGammaOWA"),
	};

	/**
	 * Type of measure to use
	 */
	int m_Type = LOWER; // default to lower approximation

	public void setType(SelectedTag newType) {
		if (newType.getTags() == TAGS_TYPE) {
			m_Type = newType.getSelectedTag().getID();
		}
	}

	public SelectedTag getType() {
		return new SelectedTag(m_Type, TAGS_TYPE);
	}
	
	public class Index implements Comparable<Index>{
		int object;
		double sim=0;

		public Index(int a, double g) {
			object = a;
			sim = g;
		}

		public int object() {
			return object;
		}

		//sort in descending order (best first)
		public int compareTo(Index o) {
			return Double.compare(o.sim,sim);
		}

		public String toString() {
			return object+":"+sim;
		}

	}

	public LAIR() {
		try {
			m_Similarity.setTNorm(new TNormKD());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * Returns a string describing this classifier
	 * @return a description of the classifier suitable for
	 * displaying in the explorer/experimenter GUI
	 */
	public String globalInfo() {
		return "Fuzzy-rough instance selection, using the decision classes and foresets.";
	}

	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(5);

		newVector.addElement(new Option(
				"\t.",
				"C", 1, "-C <num>"));


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

		optionString = Utils.getOption('I', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid Implicator specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setImplicator( (Implicator) Utils.forName( Implicator.class, className, moreOptions) );
		}
		else {
			setImplicator(new ImplicatorLukasiewicz());
		}


		optionString = Utils.getOption('T', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid TNorm specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setTNorm( (TNorm) Utils.forName( TNorm.class, className, moreOptions) );
		}
		else {
			setTNorm(new TNormLukasiewicz());
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
			setSimilarity(new Similarity3());
		}

		String thresh = Utils.getOption('K', options);
		if (thresh.length() != 0) {
			setThreshold(Double.valueOf(thresh));
		} else {
			setThreshold(1);
		}

		
		// set the type of measure to use
		String type = Utils.getOption('C', options);
		if (type.length() != 0) {
			m_Type = Integer.parseInt(type);
		} else {
			m_Type = LOWER;
		}
		
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

	public void setThreshold(double t) {
		m_threshold = t;
	}

	/**
	 * Gets the current settings of FuzzyRoughSubsetEval
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions ()  {
		Vector<String>	result;

		result = new Vector<String>();

		result.add("-I");
		result.add((m_Implicator.getClass().getName() + " " +
				Utils.joinOptions(m_Implicator.getOptions())).trim());

		result.add("-T");
		result.add((m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim());

		result.add("-R");
		result.add((m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim());

		result.add("-K");
		result.add(String.valueOf(getThreshold()));


		result.add("-C");
		result.add(String.valueOf(m_Type));

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

			int m_numClasses = m_trainInstances.numClasses();
			m_Similarity.setInstances(m_trainInstances);
			m_DecisionSimilarity.setInstances(m_trainInstances);
			m_SimilarityEq.setInstances(m_trainInstances);
			m_composition = m_Similarity.getTNorm();

			//setup the similarity matrix
			double[][] sims = new double[m_numInstances][m_numInstances];

			for (int x=0;x<m_numInstances;x++) {
				sims[x][x]=1;

				for (int y=x+1;y<m_numInstances;y++) {
					double sim = 1;		//we're t-norming				
					//double sim = 0; 

					//calculate the similarity of x and y
					for (int a=0;a<m_numAttribs;a++) {
						if (a!=m_classIndex) {
							double mainVal=m_trainInstances.instance(x).value(a);
							double otherVal=m_trainInstances.instance(y).value(a);
							double calc=0;

							if (m_trainInstances.attribute(a).isNumeric()) calc = m_Similarity.similarity(a, mainVal, otherVal);
							else calc = m_SimilarityEq.similarity(a, mainVal, otherVal);

							sim = m_composition.calculate(sim, calc);
							//sim += calc*calc;
						}
					}
					//sim /= m_numAttribs;
					sims[x][y] = sims[y][x] = sim;
				}

			}

			//each instance will have one approximation membership (to its own decision class)
			double[] lowers = new double[m_numInstances];

			//an array of ArrayLists - one ArrayList per decision class to store lower approximations and objects
			ArrayList<Index>[] perDecision = (ArrayList<Index>[])new ArrayList[m_numClasses];

			//initialise these
			for (int d=0;d<m_numClasses;d++) {
				perDecision[d] = new ArrayList<Index>();
			}


			//calculate lower approximations
			for (int x=0;x<m_numInstances;x++) {
				//get the class index
				int x_dec = (int) m_trainInstances.instance(x).classValue();

				switch(m_Type) {
				case LOWER:
					lowers[x]=1; //we're using min

					for (int y=0;y<m_numInstances;y++) {
						int dec=1;
						if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

						lowers[x] = Math.min(lowers[x], m_Implicator.calculate(sims[x][y],dec));
					}
					break;
				case VQ:
					double val = 0;
					double denom = 0;

					// for each fuzzy equivalence class in the relation
					for (int y = 0; y < m_numInstances; y++) {
						int dec=1;
						if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;
						
						double condSim = sims[x][y];
						val += m_TNorm.calculate(condSim, dec);

						denom += condSim;

					}
					lowers[x] = Q(val / denom, alpha, beta);
					break;
				case LOWER_OWA:
					PriorityQueue<Double> implications = new PriorityQueue<Double>();
					boolean modification = true;
					
					for (int y = 0; y < m_numInstances; y++) {
						int dec=1;
						if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;
						
						double condSim = sims[x][y];

						if (modification) {
							if (dec!=1) implications.add(m_Implicator.calculate(condSim, dec));
						}
						else implications.add(m_Implicator.calculate(condSim, dec));							
					}

					double sum = 0;
					int i = 0;
					int total = implications.size();

					for(Double implication: implications){							
						sum += implication*weight(total-i,alpha,beta,total);
						i++;
					}

					lowers[x] = sum;
					break;
				}
				
				perDecision[x_dec].add(new Index(x,lowers[x]));
			}

			
			//sort these with larger lower approximation memberships first
			for (int d=0;d<m_numClasses;d++) {
				Collections.sort(perDecision[d]);
			}

			BitSet instancesToRemove = new BitSet(m_numInstances);
			BitSet considered = new BitSet(m_numInstances);

			//main algorithm for instance selection
			for (int d=0;d<m_numClasses;d++) {

				//iterate over all objects belonging to the class d
				for (Index objectX: perDecision[d]) {
					int x = objectX.object;
					

					//if this object has been marked for removal, ignore it
					if (!considered.get(x)&&!instancesToRemove.get(x)) {
						considered.set(x);

						//remove all instances that are covered by instance x
						for (Index objectY: perDecision[d]) {
							int y = objectY.object; //i2.sim contains the lower approximation membership of i2 to d

							//if the threshold < 0 then use the lower approximation memberships as the cutoff point
							//(take the negation of the lower approx membership so that objects with smaller membership have less coverage)
							//otherwise use the user-supplied threshold value
							if (!considered.get(y)) {
								if (m_threshold<0&&sims[x][y]>=(1-objectY.sim)) instancesToRemove.set(y);
								//if (m_threshold<0&&sims[x][y]>=(i2.sim)) instancesToRemove.set(y);
								else if (m_threshold>=0&&sims[x][y]>m_threshold) instancesToRemove.set(y);
							}
						}

					}
				}
			}

			System.out.println(considered.cardinality());

			//remove the redundant instances (check to see if they've been flagged for removal
			setOutputFormat(m_trainInstances);
			int count=0;

			for (int i = 0; i < getInputFormat().numInstances(); i++) {
				if (instancesToRemove.get(i)) count++;
				else push(getInputFormat().instance(i));
			}
			removed = count;

			System.err.println("Removed "+removed+" instances");
		}

		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;
		return (numPendingOutput() != 0);
	}

	public double alpha=0.2;
	public double beta=1;
	
	// fuzzy quantifier
	private final double Q(double x, double alpha, double beta) {
		double ret = 0;
		double denomVal = (beta - alpha) * (beta - alpha);

		if (x <= alpha)
			ret = 0;
		else if (x < ((alpha + beta) / 2)) {
			ret = (2 * (x - alpha) * (x - alpha)) / denomVal;
		} else if (x < beta) {
			ret = 1 - ((2 * (x - beta) * (x - beta)) / denomVal);
		} else if (beta <= x) {
			ret = 1;
		}

		return ret;

	}
	
	private double weight(double i, double a, double b, double n){		
		return (Q(i/n,a,b)-Q((i-1)/n,a,b));

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
		return RevisionUtils.extract("$Revision: 5500 $");
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain arguments to the filter: 
	 * use -h for help
	 */
	public static void main(String [] argv) {
		runFilter(new LAIR(), argv);
	}

	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.compareToIgnoreCase("measureNumInstancesRemoved") == 0) {
			return measureNumInstancesRemoved();
		} else {
			throw new IllegalArgumentException(additionalMeasureName 
					+ " not supported (LAIR)");
		}
	}

	//the number of instances removed by the filter
	double removed=0;

	private double measureNumInstancesRemoved() {
		return removed;
	}

	@Override
	public Enumeration<String> enumerateMeasures() {
		Vector<String> newVector = new Vector<String>(3);
		newVector.addElement("measureNumInstancesRemoved");
		return newVector.elements();
	}
}
