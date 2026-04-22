package weka.filters.supervised.instance;


import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;

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
 *    Copyright (C) 2023 Richard Jensen
 *
 */



import weka.core.Capabilities;
import weka.core.DenseInstance;
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
import weka.fuzzy.implicator.*;
import weka.fuzzy.similarity.*;
import weka.fuzzy.snorm.*;
import weka.fuzzy.tnorm.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Enumeration;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * Fuzzy-rough oversampling
 * <p/>
 <!-- globalinfo-end -->
 * 
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 <!-- options-end -->
 *
 * @author Richard Jensen
 * @version $Revision: 1.1 $
 */
public class FROversampleBasic 
extends Filter
implements SupervisedFilter, OptionHandler, Serializable, AdditionalMeasureProducer {

	/** for serialization */
	static final long serialVersionUID = 4752870393679263361L;

	public Similarity m_Similarity = new Similarity1();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity = new SimilarityEq();

	/** the random seed to use. */
	protected int m_RandomSeed = 1;

	public TNorm m_TNorm = new TNormLukasiewicz();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();

	public boolean useAverage = true; // use the average of the lower approximation memberships

	/** the various measure types */
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
		double measure=0;
		double upr=0;
		double[] upr_array;

		public Index(int a, double g) {
			object = a;
			measure = g;
		}

		public Index(int a, double g, double u) {
			object = a;
			measure = g;
			upr = u;
		}

		public Index(int a, double g, double u, double uprArray[]) {
			object = a;
			measure = g;
			upr = u;
			upr_array = uprArray;
		}

		public int object() {
			return object;
		}

		//sort in descending order (best first)
		public int compareTo(Index o) {
			return Double.compare(o.measure,measure);
		}

		public String toString() {
			return object+":"+measure+" "+upr;
		}

	}

	public FROversampleBasic() {
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
		return "Fuzzy-rough oversampling.";
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

		String percentageStr = Utils.getOption('P', options);
		if (percentageStr.length() != 0) {
			setPercentage(Double.valueOf(percentageStr));
		} else {
			setPercentage(100.0);
		}

		String seedStr = Utils.getOption('S', options);
		if (seedStr.length() != 0) {
			setRandomSeed(Integer.parseInt(seedStr));
		} else {
			setRandomSeed(1);
		}

		setUseAverage(Utils.getFlag('A', options));

		setNoiseRemoval(Utils.getFlag('N', options));


		// set the type of measure to use
		String type = Utils.getOption('C', options);
		if (type.length() != 0) {
			m_Type = Integer.parseInt(type);
		} else {
			m_Type = LOWER;
		}
	}

	public void setUseAverage(boolean ua) {
		useAverage = ua;
	}

	public boolean getUseAverage() {
		return useAverage;
	}

	public String useAverageTipText() {
		return "Use the average of the measure (e.g. the lower approximation) as a threshold rather than the lowest lower approximation membership.";
	}


	public void setNoiseRemoval(boolean nr) {
		noiseRemoval = nr;
	}

	public boolean getNoiseRemoval() {
		return noiseRemoval;
	}

	public String noiseRemovalTipText() {
		return "Not implemented currently but may be in the future.";
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

	public double getPercentage() {
		return percentToGenerate;
	}

	public void setPercentage(double p) {
		if (p<=0 || p>100) p=100;
		percentToGenerate = p;
	}


	/**
	 * Returns the tip text for this property.
	 * 
	 * @return 		tip text for this property suitable for
	 * 			displaying in the explorer/experimenter gui
	 */
	public String percentageTipText() {
		return "The percentage of instances to create.";
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return 		tip text for this property suitable for
	 * 			displaying in the explorer/experimenter gui
	 */
	public String randomSeedTipText() {
		return "The seed used for random sampling.";
	}


	/**
	 * Gets the random number seed.
	 *
	 * @return 		the random number seed.
	 */
	public int getRandomSeed() {
		return m_RandomSeed;
	}

	/**
	 * Sets the random number seed.
	 *
	 * @param value 	the new random number seed.
	 */
	public void setRandomSeed(int value) {
		m_RandomSeed = value;
	}


	/**
	 * Gets the current settings
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions ()  {
		Vector<String> result;

		result = new Vector<String>();

		result.add("-I");
		result.add((m_Implicator.getClass().getName() + " " +
				Utils.joinOptions(m_Implicator.getOptions())).trim());

		result.add("-T");
		result.add((m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim());

		result.add("-R");
		result.add((m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim());

		result.add("-P");
		result.add(String.valueOf(getPercentage()));

		result.add("-S");
		result.add("" + getRandomSeed());

		if (getUseAverage()) result.add("-A");

		if (getNoiseRemoval()) result.add("-N");

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

	boolean noiseRemoval=false; // not currently used but could be added
	double percentToGenerate = 100; // what percentage of instances to generate so that the total will match the majority class size

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


			//set up the similarity matrix
			double[][] sims = new double[m_numInstances][m_numInstances];

			for (int x=0;x<m_numInstances;x++) {
				sims[x][x]=1;

				for (int y=x+1;y<m_numInstances;y++) {
					double sim = getInstanceSimilarity(m_trainInstances.get(x), m_trainInstances.get(y), m_numAttribs, m_classIndex, m_trainInstances);
					
					//sim /= m_numAttribs;
					sims[x][y] = sims[y][x] = sim;
				}

			}

			// for each class, record the lowest lower approx membership to use as a threshold later
			// might be better using the average membership as the threshold
			double[] lowestLowerApprox = new double[m_numClasses];
			double[][] x_upper = new double[m_numClasses][m_numAttribs]; // range of attribute values per decision class
			double[][] x_lower = new double[m_numClasses][m_numAttribs];

			// initialise the arrays to the correct values
			for (int d = 0; d < m_numClasses; d++) {
				for (int a=0;a<m_numAttribs;a++) {
					if (a==m_classIndex) continue;
					x_upper[d][a] = Double.MIN_VALUE;
					x_lower[d][a] = Double.MAX_VALUE;
				}

				if (useAverage) lowestLowerApprox[d] = 0;
				else lowestLowerApprox[d] = 1;
			}


			// determine the majority class
			int majorityClass = 0;
			int majoritySize = Integer.MIN_VALUE;

			int[] classCounts = getInputFormat().attributeStats(getInputFormat().classIndex()).nominalCounts;
			for (int d = 0; d < classCounts.length; d++) {
				if (classCounts[d] > majoritySize) {
					majoritySize = classCounts[d];
					majorityClass = d;
				}
			}

			BitSet instancesToRemove = new BitSet(m_numInstances); // not currently used

			for (int i=0;i<m_numInstances;i++) {
				//get the class index
				int decisionClass = (int) m_trainInstances.instance(i).classValue();
				if (majorityClass==decisionClass) continue;

				double[] approx = approximations(i,decisionClass,m_numInstances,sims,m_trainInstances,instancesToRemove);

				if (useAverage) lowestLowerApprox[decisionClass]+=approx[0];
				else if (approx[0]<lowestLowerApprox[decisionClass]) lowestLowerApprox[decisionClass]=approx[0];
				
				for (int a=0;a<m_numAttribs;a++) {
					if (a==m_classIndex) continue;

					double value = m_trainInstances.get(i).value(a);
					x_upper[decisionClass][a] = Math.max(x_upper[decisionClass][a], value);
					x_lower[decisionClass][a] = Math.min(x_lower[decisionClass][a], value);
				}
			}

			if (useAverage) {
				for (int d = 0; d < classCounts.length; d++) {
					lowestLowerApprox[d]/=classCounts[d];
				}
			}



			//BitSet considered = new BitSet(m_numInstances);
			Random rand = new Random(getRandomSeed());

			if (noiseRemoval) { // not used but may be later

			}

			added=0; // keeps track of the number of added instances
			for (int i=0; i<m_numInstances; i++) push(m_trainInstances.get(i)); // add the original data

			int NUM_ATTEMPTS = 10; //number of times to try to create an instance before giving up (avoid potential infinite looping)
			
			//main algorithm for instance creation
			for (int d=0; d < m_numClasses; d++) {
				if (d==majorityClass) continue; // ignore the majority class

				int numToGenerate = (int) Math.floor(((percentToGenerate/100)*majoritySize)-classCounts[d]);
				double threshold = lowestLowerApprox[d];
				double attempts = NUM_ATTEMPTS;
				
				// generate the new instances
				// use the lowest or average lower approximation membership for all instances that belong to class d
				// to decide if the generated instance should be kept
				for (int i=0; i<numToGenerate; i++) {

					Enumeration<Attribute> attrEnum = getInputFormat().enumerateAttributes();
					double[] values = new double[m_numAttribs];

					// create the new instance
					// could the lower approx membership be used to weight these somehow?
					while(attrEnum.hasMoreElements()) {
						Attribute attr = (Attribute) attrEnum.nextElement();
						int a = attr.index();

						// if not the class
						if (!attr.equals(getInputFormat().classAttribute())) {
							if (attr.isNumeric()) {
								// numeric: 
								values[attr.index()] = rand.nextDouble()*(x_upper[d][a]-x_lower[d][a])+x_lower[d][a];
							} else {							
								// nominal: 
								int numValues = attr.numValues();

								values[attr.index()] = rand.nextInt(numValues);
							}
						}
					}

					values[m_classIndex] = d; // set the class of the new instance to d
					Instance synthetic = new DenseInstance(1.0, values);
					double[] eval = evaluate(synthetic, d,m_numInstances,m_numAttribs,m_classIndex,m_trainInstances);
					
					//System.err.println(eval[0]+" "+threshold);
					
					if (eval[0]<threshold) { // a weak instance has been created
						//System.err.println("Weak instance, attempts:"+attempts);
						if (attempts>0) {i--;attempts--;} // if there are still attempts left, then reiterate
						else {attempts = NUM_ATTEMPTS;} // otherwise, give up at this point
					}
					else {
						push(synthetic); // add the instance
						added++;
						attempts = NUM_ATTEMPTS; // reset the counter
					}
				}
			}
		}

		System.err.println("added "+added+" instances");
		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;
		return (numPendingOutput() != 0);
	}
	
	/**
	 * Calculates the fuzzy similarity of two instances
	 * @param instanceX First instance
	 * @param instanceY Second instance
	 * @param m_numAttribs Number of attributes in the data
	 * @param m_classIndex Index of the class attribute
	 * @param m_trainInstances The training instances
	 * @return the fuzzy similarity
	 */
	private double getInstanceSimilarity(Instance instanceX, Instance instanceY, int m_numAttribs, int m_classIndex, Instances m_trainInstances) {
		double sim = 1;		//we're t-norming	
		
		//calculate the similarity of x and y
		for (int a=0;a<m_numAttribs;a++) {
			if (a!=m_classIndex) {
				double mainVal=instanceX.value(a);
				double otherVal=instanceY.value(a);
				double calc=0;

				if (m_trainInstances.attribute(a).isNumeric()) calc = m_Similarity.similarity(a, mainVal, otherVal);
				else calc = m_SimilarityEq.similarity(a, mainVal, otherVal);

				sim = m_composition.calculate(sim, calc);
				if (sim==0) break; // optimisation
				//sim += calc*calc;
			}
		}
		
		return sim;
	}


	/**
	 * Calculate the lower/upper approximations
	 * @param x the object index
	 * @param x_dec the decision class
	 * @param m_numInstances the number of instances in the data
	 * @param sims the precalculated similarity relation
	 * @param m_trainInstances the training data
	 * @param toRemove the instances that have been flagged to be removed
	 * @return (depends on the selected measure type) the lower and upper approximation membership of x
	 */
	private double[] approximations(int x, int x_dec, int m_numInstances, double[][] sims, Instances m_trainInstances, BitSet toRemove) {
		double[] ret = new double[2];

		switch(m_Type) {
		case LOWER:
			ret[0]=1; //we're using min...
			ret[1]=0; // and max too

			for (int y=0; y<m_numInstances; y++) {
				if (toRemove.get(y)) continue;

				int dec=1;
				if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

				ret[0] = Math.min(ret[0], m_Implicator.calculate(sims[x][y],dec));						
				//calc uprApproxes-[x].
				if (x!=y) ret[1] = Math.max(ret[1], m_TNorm.calculate(sims[x][y], dec));

			}
			break;

		case COMBINED:
			ret[0]=1; //we're using min...
			ret[1]=0; // and max too

			for (int y=0; y<m_numInstances; y++) {
				if (toRemove.get(y)) continue;

				int dec=1;
				if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

				ret[0] = Math.min(ret[0], m_Implicator.calculate(sims[x][y],dec));						
				//calc uprApproxes-[x].
				if (x!=y) ret[1] = Math.max(ret[1], m_TNorm.calculate(sims[x][y], dec));
				//System.out.println("uppers for " + x + " = " + uppers[x]);

			}				
			break;

		case UPPER:
			ret[0]=1; //we're using min...
			ret[1]=0; // and max too!

			for (int y=0; y<m_numInstances; y++) {
				if (toRemove.get(y)) continue;

				int dec=1;

				ret[0] = Math.min(ret[0], m_Implicator.calculate(sims[x][y],dec));
				//calc uprApproxes-[x].
				if (x!=y) ret[1] = Math.max(ret[1], m_TNorm.calculate(sims[x][y], dec));


			}
			break;

		case VQ:
			double val = 0;
			double denom = 0;

			// for each fuzzy equivalence class in the relation
			for (int y = 0; y < m_numInstances; y++) {
				if (toRemove.get(y)) continue;

				int dec=1;
				if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

				double condSim = sims[x][y];
				val += m_TNorm.calculate(condSim, dec);

				denom += condSim;

			}
			ret[0] = Q(val / denom, alpha_l, beta_l);
			ret[1] = Q(val / denom, alpha_u, beta_u);

			break;
		case LOWER_OWA:
			PriorityQueue<Double> implications = new PriorityQueue<Double>();
			PriorityQueue<Double> tnorms = new PriorityQueue<Double>();

			for (int y = 0; y < m_numInstances; y++) {
				if (toRemove.get(y)) continue;

				int dec=1;
				if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

				double condSim = sims[x][y];

				implications.add(m_Implicator.calculate(condSim, dec));
				tnorms.add(m_TNorm.calculate(condSim, dec));
			}

			double sum = 0;
			int i = 0;
			int total = implications.size();

			for (Double implication: implications){							
				sum += implication*weight(total-i,alpha_l,beta_l,total);
				i++;
			}
			ret[0] = sum;

			sum = 0; i = 0;
			total = tnorms.size();

			for (Double tnorm: tnorms) {
				sum += tnorm*weight(total-i,alpha_u,beta_u,total);
			}
			ret[1] = sum;

			break;
		}

		return ret;
	}
	
	
	/**
	 * Evaluate a newly created instance
	 * @param instance The new instance
	 * @param x_dec The class of the new instance
	 * @param m_numInstances The number of instances in the original data
	 * @param m_numAttribs The number of features/attributes
	 * @param m_classIndex The index of the class attribute
	 * @param m_trainInstances The set of training instances
	 * @return The lower and/or upper approximation memberships for the new instance
	 */
	private double[] evaluate(Instance instance, int x_dec, int m_numInstances, int m_numAttribs, int m_classIndex, Instances m_trainInstances) {
		double[] ret = new double[2];
        double[] sims = new double[m_numInstances];
        
		// generate the new similarity relation
		for (int y=0; y<m_numInstances; y++) {
            sims[y] = getInstanceSimilarity(instance, m_trainInstances.get(y), m_numAttribs, m_classIndex, m_trainInstances);
		}
		
		switch(m_Type) {
		case LOWER:
			ret[0]=1; //we're using min...
			ret[1]=0; // and max too

			for (int y=0; y<m_numInstances; y++) {

				int dec=1;
				if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

				ret[0] = Math.min(ret[0], m_Implicator.calculate(sims[y],dec));						
				//calc uprApproxes-[x].
				ret[1] = Math.max(ret[1], m_TNorm.calculate(sims[y], dec));

			}
			break;

		case COMBINED:
			ret[0]=1; //we're using min...
			ret[1]=0; // and max too

			for (int y=0; y<m_numInstances; y++) {

				int dec=1;
				if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

				ret[0] = Math.min(ret[0], m_Implicator.calculate(sims[y],dec));						
				ret[1] = Math.max(ret[1], m_TNorm.calculate(sims[y], dec));
				//System.out.println("uppers for " + x + " = " + uppers[x]);

			}				
			break;

		case UPPER:
			ret[0]=1; //we're using min...
			ret[1]=0; // and max too!

			for (int y=0; y<m_numInstances; y++) {

				int dec=1;

				ret[0] = Math.min(ret[0], m_Implicator.calculate(sims[y],dec));
				ret[1] = Math.max(ret[1], m_TNorm.calculate(sims[y], dec));


			}
			break;

		case VQ:
			double val = 0;
			double denom = 0;

			// for each fuzzy equivalence class in the relation
			for (int y = 0; y < m_numInstances; y++) {

				int dec=1;
				if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

				double condSim = sims[y];
				val += m_TNorm.calculate(condSim, dec);

				denom += condSim;

			}
			ret[0] = Q(val / denom, alpha_l, beta_l);
			ret[1] = Q(val / denom, alpha_u, beta_u);

			break;
		case LOWER_OWA:
			PriorityQueue<Double> implications = new PriorityQueue<Double>();
			PriorityQueue<Double> tnorms = new PriorityQueue<Double>();

			for (int y = 0; y < m_numInstances; y++) {

				int dec=1;
				if (x_dec!=(int)m_trainInstances.instance(y).classValue()) dec=0;

				double condSim = sims[y];

				implications.add(m_Implicator.calculate(condSim, dec));
				tnorms.add(m_TNorm.calculate(condSim, dec));
			}

			double sum = 0;
			int i = 0;
			int total = implications.size();

			for (Double implication: implications){							
				sum += implication*weight(total-i,alpha_l,beta_l,total);
				i++;
			}
			ret[0] = sum;

			sum = 0; i = 0;
			total = tnorms.size();

			for (Double tnorm: tnorms) {
				sum += tnorm*weight(total-i,alpha_u,beta_u,total);
			}
			ret[1] = sum;

			break;
		}

		return ret;
	}

	public double alpha_l=0.2;
	public double beta_l=1;
	public double alpha_u=0.1;
	public double beta_u=0.6;

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
		runFilter(new FROversampleBasic(), argv);
	}

	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.compareToIgnoreCase("measureNumInstancesAdded") == 0) {
			return measureNumInstancesAdded();
		} else {
			throw new IllegalArgumentException(additionalMeasureName 
					+ " not supported");
		}
	}

	//the number of instances removed by the filter
	double added=0;

	private double measureNumInstancesAdded() {
		return added;
	}

	@Override
	public Enumeration<String> enumerateMeasures() {
		Vector<String> newVector = new Vector<String>(3);
		newVector.addElement("measureNumInstancesAdded");
		return newVector.elements();
	}
}
