package weka.filters.supervised.instance;

// old version of LAIR2

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
import weka.estimators.MultivariateGaussianEstimator;
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
public class FRIS 
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
	
	public double m_uThreshold= -1.0; //threshold for removing objects using upper approx preprocessing 

	public TNorm m_TNorm = new TNormLukasiewicz();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();

	public boolean densityEstimation=false;

	/** the various link types */
	final static int LOWER = 0;
	final static int COMBINED = 1;
	final static int UPPER = 2;
	final static int VQ = 3;
	final static int LOWER_OWA = 4;

	public static final Tag[] TAGS_TYPE = {
			new Tag(LOWER, "WeakGamma"),
			new Tag(COMBINED, "Upper/Lower Weak Gamma"),
			new Tag(UPPER, "Combined Classes Upper Approx Weak Gamma"),
			new Tag(VQ, "VQRS"),
			new Tag(LOWER_OWA, "WeakGammaOWA"),
	};

	/**
	 * Holds the Link type used calculate distance between clusters
	 */
	int m_Type = LOWER;

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
		double upr=0;
		double [] upr_array;

		public Index(int a, double g) {
			object = a;
			sim = g;
		}
		
		public Index(int a, double g, double u) {
			object = a;
			sim = g;
			upr = u;
		}
		
		public Index(int a, double g, double u, double uprArray[]) {
			object = a;
			sim = g;
			upr = u;
			upr_array = uprArray;
		}

		public int object() {
			return object;
		}

		//sort in descending order (best first)
		public int compareTo(Index o) {
			return Double.compare(o.sim,sim);
		}

		public String toString() {
			return object+":"+sim+" "+upr;
		}

	}

	public FRIS() {
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

		String knnString = Utils.getOption('K', options);
		if (knnString.length() != 0) {
			setThreshold(Double.valueOf(knnString));
		} else {
			setThreshold(1);
		}

		setDensityEstimation(Utils.getFlag('B', options));
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

	public double getUprThreshold() {
		return m_uThreshold;
	}

	public void setUprThreshold(double u) {
		m_uThreshold = u;
	}
	
	public boolean getDensityEstimation() {
		return densityEstimation;
	}

	public void setDensityEstimation(boolean b) {
		densityEstimation=b;
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
		
		result.add("-U");
		result.add(String.valueOf(getUprThreshold()));

		if (getDensityEstimation()) {
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
			boolean combined = false;
			boolean upr =false;

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
			double[] uppers = new double[m_numInstances];
			
			

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
				double [] uprAppxArray = new double[m_numClasses];
				switch(m_Type) {
				case LOWER:
					lowers[x]=1; //we're using min...
					uppers[x]=0; // and max too

					for (int y=0;y<m_numInstances;y++) {
						int dec=1;
						if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

						lowers[x] = Math.min(lowers[x], m_Implicator.calculate(sims[x][y],dec));						
						//calc uprApproxes-[x].
						if (x!=y) uppers[x] = Math.max(uppers[x], m_TNorm.calculate(sims[x][y], dec));
														
					}
					break;
					
				case COMBINED:
					lowers[x]=1; //we're using min...
					uppers[x]=0; // and max too
					combined = true;
					for (int y=0;y<m_numInstances;y++) {
						int dec=1;
						if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;
						lowers[x] = Math.min(lowers[x], m_Implicator.calculate(sims[x][y],dec));						
						//calc uprApproxes-[x].
						if (x!=y) uppers[x] = Math.max(uppers[x], m_TNorm.calculate(sims[x][y], dec));
						//System.out.println("uppers for " + x + " = " + uppers[x]);
																	
					}				
					break;
					
				case UPPER:
					lowers[x]=1; //we're using min...
					uppers[x]=0; // and max too!
					upr = true;
					 
					
					for (int y=0;y<m_numInstances;y++) {
						int dec=1;
						int y_dec = (int) m_trainInstances.instance(y).classValue();
						
						//if y is of a different class then make dec=0 so that the lwr calc is only for the same class
						if (x_dec!=y_dec) {
							dec=0;
							//populate the arrays with the upr approx values of *other* classes
							for(int decClass=0;decClass<m_numClasses;decClass++){
							if(decClass == y_dec) uprAppxArray[y_dec] = Math.max(uprAppxArray[y_dec], m_TNorm.calculate(sims[x][y], 1));							
							else uprAppxArray[decClass] = Math.max(uprAppxArray[decClass], m_TNorm.calculate(sims[x][y], 0));
							}
						}
					

						lowers[x] = Math.min(lowers[x], m_Implicator.calculate(sims[x][y],dec));
						//calc uprApproxes-[x].
						if (x!=y) uppers[x] = Math.max(uppers[x], m_TNorm.calculate(sims[x][y], dec));
																									
						
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
				//combined lwr and upr-[x] for a given class
				if (combined) {
					lowers[x] = (lowers[x] + uppers[x])/2;
					perDecision[x_dec].add(new Index(x,lowers[x],uppers[x]));
				}
				
				//Upper approxes for each class 
				else if (upr){
										
					if(lowers[x] != 1.0) lowers[x] = (lowers[x] + uppers[x])/2;
								
					//get the upper approx memberships from every other class
					int cnt=0;
					double mean = 0;
					for(int t=0; t<m_numClasses; t++){
						if(t==x_dec) continue;
						double uprMemb = uprAppxArray[t];
										
						if (uprMemb > 0) {
							mean+= uprMemb;
							cnt++;
						}											
					}
					mean = mean/cnt;
					if (uppers[x] == 0) lowers[x] = 0;									
				perDecision[x_dec].add(new Index(x,lowers[x],uppers[x], uprAppxArray));
					
				}
				//else do the normal LAIR
				else perDecision[x_dec].add(new Index(x,lowers[x],uppers[x]));

			}

			//only consider the density on a per-class basis
			MultivariateGaussianEstimator[] mge = new MultivariateGaussianEstimator[m_numClasses];

			if (densityEstimation) {
				for (int d=0;d<m_numClasses;d++) {
					//initialise
					mge[d] = new MultivariateGaussianEstimator();

					//TODO: will need to remove the class attribute from consideration...
					double[][] obs = new double[perDecision[d].size()][m_numAttribs-1];
					int i = 0;

					//get all instances in class d
					for (Index i2: perDecision[d]) {
						double[] data = m_trainInstances.instance(i2.object).toDoubleArray();
						int attrIndex=0;

						for (int a=0;a<m_numAttribs;a++) {
							if (a!=m_classIndex) {
								obs[i][attrIndex] = data[a];
								attrIndex++;
							}
						}
						i++;
					}

					mge[d].estimate(obs,null);
				}


			}

			//sort these with larger lower or combined lower/upper approximation memberships first
			for (int d=0;d<m_numClasses;d++) {
				Collections.sort(perDecision[d]);
			}
			
			for (int d=0;d<m_numClasses;d++) {
			for (Index i: perDecision[d]) {
				//System.out.println("metric = "+ i.sim +" for object: " + i.object);
			}
			}
			

			BitSet instancesToRemove = new BitSet(m_numInstances);
			BitSet considered = new BitSet(m_numInstances);
			int cntr =0;
			

			//main algorithm for instance selection
			for (int d=0;d<m_numClasses;d++) {

				//iterate over all objects belonging to the class d
				for (Index i: perDecision[d]) {
					int x = i.object;
					
//					//If modified upper approx membership (upr-[x]) for x is below threshold, then mark for removal as a preprocessing step
 					if (m_uThreshold > 0 && i.upr < m_uThreshold){
					instancesToRemove.set(x);
					cntr++;	
					continue;
					}
					
 					//If the upper approxes are used to select instances then remove objects with lowers = 0
					if(upr) {
					if (i.sim == 0) instancesToRemove.set(x);
					}
					
										

					//if this object has been marked for removal, ignore it
					if (!considered.get(x)&&!instancesToRemove.get(x)) {
						considered.set(x);

						//remove all instances that are covered by instance x 
						for (Index i2: perDecision[d]) {
							int y = i2.object; //i2.sim contains the lower approximation membership of i2 to d

							//if the threshold < 0 then use the lower approximation memberships as the cutoff point
							//(take the negation of the lower approx membership so that objects with smaller membership have less coverage)
							//otherwise use the user-supplied threshold value
							if (!considered.get(y)) {
								if (m_threshold<0&&sims[x][y]>=(1-i2.sim)){
									instancesToRemove.set(y);
									
								}
								//if (m_threshold<0&&sims[x][y]>=(i2.sim)) instancesToRemove.set(y);
								else if (m_threshold>=0&&sims[x][y]>m_threshold) {
									instancesToRemove.set(y);
									
								}
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
			//System.err.println("Removed "+ cntr + " instances by upr approx");
			
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
		runFilter(new FRIS(), argv);
	}

	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.compareToIgnoreCase("measureNumInstancesRemoved") == 0) {
			return measureNumInstancesRemoved();
		} else {
			throw new IllegalArgumentException(additionalMeasureName 
					+ " not supported (FRIS)");
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
