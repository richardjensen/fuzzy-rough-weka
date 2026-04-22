package weka.classifiers.fuzzy;

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
 *    FuzzyRoughNN.java
 *
 */


import weka.classifiers.Classifier;
import weka.classifiers.fuzzy.FuzzyRoughRuleInducer.FuzzyRule;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.*;
import weka.fuzzy.tnorm.*;
import weka.fuzzy.snorm.*;
import weka.fuzzy.similarity.*;

import java.io.Serializable;
import java.util.*;


/**
 <!-- globalinfo-start -->
 * Fuzzy-rough nearest-neighbour classifier. <br/>
 * <br/>
 * 
 * <p/>
 <!-- globalinfo-end -->
 * 
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author Richard Jensen
 * @version $Revision: 1.19 $
 */
public class EntropyRules 
extends FuzzyRoughRuleInducer 
implements OptionHandler,
TechnicalInformationHandler {

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;
	
	//whether to use the weak positive region or not
	public boolean m_weak=true;
	public final static double LN2 = 0.69314718055994531;
	
	public EntropyRules() {
		init();
	}

	/**
	 * Returns a string describing classifier.
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return  "Fuzzy entropy-based rule induction.\n\n"
		+ "For more information, see\n\n"
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
		TechnicalInformation 	result;
		result = new TechnicalInformation(Type.CONFERENCE);
		result.setValue(Field.AUTHOR, "R. Jensen and C. Cornelis");
		result.setValue(Field.YEAR, "2008");
		result.setValue(Field.TITLE, "A New Approach to Fuzzy-Rough Nearest Neighbour Classification");
		result.setValue(Field.BOOKTITLE, "6th International Conference on Rough Sets and Current Trends in Computing");
		result.setValue(Field.PAGES, "310-319");

		return result;
	}




	/**
	 * Get the number of training instances the classifier is currently using.
	 * 
	 * @return the number of training instances the classifier is currently using
	 */
	public int getNumTraining() {

		return m_numInstances;
	}

	/**
	 * Returns default capabilities of the classifier.
	 *
	 * @return      the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}



	/**
	 * Returns an enumeration describing the available options.
	 *
	 * @return an enumeration of all the available options.
	 */
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(8);

		newVector.addElement(new Option(
				"\tThe nearest neighbour search algorithm to use "+
				"(default: weka.core.neighboursearch.LinearNNSearch).\n",
				"A", 0, "-A"));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options. <p/>
	 *
 <!-- options-start -->
	 * Valid options are: <p/>
	 * 
 <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {

		String optionString = Utils.getOption('T', options);
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

		setWeak(Utils.getFlag('W', options));
		
		Utils.checkForRemainingOptions(options);
	}

	public void setWeak(boolean w) {
		m_weak = w;
	}
	
	public boolean getWeak() {
		return m_weak;
	}

	//set the relation composition operator = tnorm
	public void setTNorm(TNorm tnorm) {
		m_TNorm = tnorm;
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

	/**
	 * Gets the current settings of IBk.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String [] getOptions() {

		String [] options = new String [15];
		int current = 0;

		options[current++] = "-T";
		options[current++] = (m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim();

		options[current++] = "-R";
		options[current++] = (m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim(); 

		if (getWeak()) {
			options[current++] = "-W";
		}
		
		while (current < options.length) {
			options[current++] = "";
		}

		return options;
	}




	/**
	 * Returns a description of this classifier.
	 *
	 * @return a description of this classifier as a string.
	 */
	public String toString() {

		if (m_Train == null) {
			return "QuickRules: No model built yet.";
		}

		String result = "QuickRules\n" +
		"Fuzzy-rough rule induction\n";
		StringBuffer text = new StringBuffer();
		text.append("\nSimilarity measure: "+m_Similarity);
		text.append("\nT-Norm: "+m_TNorm+"\nRelation composition: "+m_Similarity.getTNorm());
		text.append("\n\nNumber of rules: "+cands.size());
		text.append("\nAverage rule arity: "+arity);

		result+=text.toString();

		return result;
	}

	/**
	 * Initialise scheme variables.
	 */
	protected void init() {
		m_TNorm = new TNormLukasiewicz();
		m_Similarity = new Similarity1();

	}

	public double decCardinality=0;
	public double[] decisionLength;
	public double[] cards;
	
	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data 
	 * @throws Exception if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

		m_numClasses = instances.numClasses();
		m_ClassType = instances.classAttribute().type();
		m_Train = new Instances(instances, 0, instances.numInstances());

		m_Similarity.setInstances(m_Train);

		m_composition = m_Similarity.getTNorm();
		m_SNorm = m_TNorm.getAssociatedSNorm();
		m_classIndex = m_Train.classIndex();
		m_numInstances = m_Train.numInstances();

		globallowers = new double[m_numInstances];
		global = new double[m_numInstances];
		
		arity=0;
		current = new Relation(m_numInstances);
		temprel = new Relation(m_numInstances);
		initCurrent(); 

		decisionLength = new double[m_numInstances];
		cards = new double[m_numInstances];
		
		for (int x = 0; x < m_numInstances; x++) {
			for (int y = 0; y < m_numInstances; y++) {
				decisionLength[x] += fuzzySimilarity(m_classIndex, m_Train.instance(x).value(m_classIndex), m_Train.instance(y).value(m_classIndex));
			}

			decCardinality += 1 / decisionLength[x];
		}
		
		
		fullData=true;
		//work out the consistency
		BitSet full = new BitSet(m_Train.numAttributes());
		//for (int a=0;a<m_Train.numAttributes();a++) if (a!=m_classIndex) full.set(a);
		full.set(0, m_Train.numAttributes());
		full.clear(m_classIndex);
		//generatePartition(full);
		for (int a=0;a<m_Train.numAttributes();a++) {
			if (a!=m_classIndex) {
				generatePartition(current,a);
				setCurrent(temprel);
			}
		}

		indexes = new int[m_numInstances];
		//prepare optimization information
		decIndexes = new int[m_numInstances];

		if (!m_weak) {
			for (int i = 0; i < m_numInstances; i++) {
				boolean same = false;

				for (int j = i + 1; j < m_numInstances; j++) {

					for (int d = 0; d < m_numInstances; d++) {
						double decSim1 = fuzzySimilarity(m_classIndex, m_Train.instance(i).value(m_classIndex), m_Train.instance(d).value(m_classIndex));
						double decSim2 = fuzzySimilarity(m_classIndex, m_Train.instance(j).value(m_classIndex), m_Train.instance(d).value(m_classIndex));

						if (decSim1!=decSim2) {same=false; break;}
						else same=true;

					}
					if (same) {
						decIndexes[j] = i + 1;
						break;
					} // i+1 as default is 0
				}
			}
		}

		double fullGamma;

		if (m_ClassType != Attribute.NOMINAL&&!m_weak) fullGamma = calculateEntropy2(current,full);
		else fullGamma = calculateEntropy2(current,full);

		fullData=false;

		//for (int o=0;o<m_numInstances;o++) System.err.print(globallowers[o]+" ");
		//System.err.println("full = "+fullGamma);

		induce(fullGamma);
	}


	/**
	 * Induce (and store) rules via QuickRules
	 */
	public void induce(double full) {
		initCurrent(); 
		cands = new ArrayList<FuzzyRule>(m_numInstances);
		BitSet reduct = new BitSet(m_Train.numAttributes());
		double gamma=-1;
		int bestAttr=-1;

		while ((float)gamma != (float)full) {
			bestAttr=-1;
			for (int a=0;a<m_Train.numAttributes();a++) {
				if (a != m_classIndex && !reduct.get(a)) {
					reduct.set(a);
					generatePartition(current,a);

					double gam;
					if (m_ClassType != Attribute.NOMINAL&&!m_weak) gam = calculateEntropy2(temprel,reduct);
					else gam = calculateEntropy2(temprel,reduct);
	
					//System.err.println(reduct+" => "+gam);
					reduct.clear(a);

					if (gam>=gamma) {
						gamma=gam;
						bestAttr=a;
					}
				}
			}

			if (bestAttr==-1)  break;

			reduct.set(bestAttr);
			
			if (m_Debug) System.err.println(reduct+" => "+(gamma/full));
			if (allCovered()) break;
			
			generatePartition(current,bestAttr);
			setCurrent(temprel);			
		}
		if (m_Debug) {
			Iterator<FuzzyRule> it = cands.iterator();
			FuzzyRule rule;

			while (it.hasNext()) {
				rule = it.next();
				arity+=rule.attributes.cardinality();
			}
			arity/=cands.size();
			System.err.println("Finished induction: "+cands.size()+" rule(s)");
			System.err.println("Average rule arity = "+arity+"\n");
		}

	}


/*	private final double calculateEntropy(Relation rel, BitSet reduct) {
		double ret=0;
		double[] eqClass; 

		double cardinality = 0; double temp;
		
		for (int x = 0; x < m_numInstances; x++) {
			temp = 0;
			int d=x;

			double val = 0;
			double denom = 0;// temp=0;
						
			eqClass = new double[m_numInstances];
						
			for (int y = 0; y < m_numInstances; y++) {
				double condSim = rel.getCell(x, y);
				val += m_TNorm.calcNorm(condSim, fuzzySimilarity(m_classIndex, m_Train.instance(d).value(m_classIndex), m_Train.instance(y).value(m_classIndex)));
				eqClass[y] = condSim;

				denom += condSim;

			}
			cards[x] = denom;
			temp += (-(val / denom) * log2(val / denom))/decisionLength[d];

						
			if (!fullData&&val==denom&&globallowers[x][0]>global[x]) {
				determineAdd(new FuzzyRule(eqClass,(BitSet)reduct.clone(),x));
			}	
			
			vals[x] = cards[x] * temp;
	
			cardinality += cards[x];
			ret += vals[x];
			if (fullData) globallowers[x][0]=vals[x]/cards[x];
		}
		ret /= cardinality;
		return 1 - (ret);
	}*/
	
	//full version
	private final double calculateEntropy2(Relation rel, BitSet reduct) {
		double ret=0;
		double[] eqClass; 
		vals = new double[m_numInstances];
		indexes = new int[m_numInstances];
		double cardinality = 0; double temp;
		setIndexes(rel);

		for (int x = 0; x < m_numInstances; x++) {
			temp = 0;
			// System.out.print(indexes[x]+" ");
			if (indexes[x] != 0) {
				cards[x] = cards[indexes[x] - 1];
				vals[x] = vals[indexes[x] - 1];
			} else {
				decVals = new double[m_numInstances];
				for (int d = 0; d < m_numInstances; d++) {
					if (decIndexes[d] != 0) {
						decVals[d] = decVals[decIndexes[d] - 1];
						temp += decVals[d];
					} else {
						double val = 0;
						double denom = 0;// temp=0;
						
						eqClass = new double[m_numInstances];
						
						for (int y = 0; y < m_numInstances; y++) {
							double condSim = rel.getCell(x, y);
							val += m_TNorm.calculate(condSim, fuzzySimilarity(m_classIndex, m_Train.instance(d).value(m_classIndex), m_Train.instance(y).value(m_classIndex)));
							eqClass[y] = condSim;

							denom += condSim;

						}
						cards[x] = denom;

						decVals[d] = (-(val / denom) * log2(val / denom)) / decisionLength[d];
						temp += decVals[d];
						
						if (!fullData&&val==denom&&globallowers[x]<global[x]) {
							determineAdd(new FuzzyRule(eqClass,(BitSet)reduct.clone(),x));
						}
					}				
				}
				
				vals[x] = cards[x] * temp;

			}
			cardinality += cards[x];
			ret += vals[x];
			if (fullData) globallowers[x]=vals[x]/cards[x];
		}
		ret /= cardinality;
		return 1 - (ret / log2(decCardinality));
	}

	public static final double log2(double x) {
		if (x > 0)
			return Math.log(x) / LN2;
		else
			return 0.0;
	}


	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if an error occurred during the prediction
	 */
	public double [] distributionForInstance(Instance instance) throws Exception {

		if (m_numInstances == 0) {
			throw new Exception("No training instances!");
		}


		double [] distribution=null;
		if (m_ClassType == Attribute.NOMINAL) distribution = new double[m_numClasses];
		else distribution = new double[m_numInstances];

		double total=0;
		double bestValue=-1;
		int bestClass=-1;

		//System.err.println(neighbours.numInstances());

		if (m_ClassType == Attribute.NOMINAL) {
			distribution = new double [m_numClasses];
			//double[] classCounts = new double[m_numClasses];

			Iterator<FuzzyRule> it = cands.iterator();
			FuzzyRule rule;

			while (it.hasNext()) {
				rule = it.next();
				double value=1;

				for (int a = rule.attributes.nextSetBit(0); a >= 0; a = rule.attributes.nextSetBit(a + 1)) {
					if (a != m_classIndex) {
						value = m_composition.calculate(fuzzySimilarity(a,m_Train.instance(rule.object).value(a),instance.value(a)), value);
						//value = m_TNorm.calcNorm(fuzzySimilarity(a,m_Train.instance(rule.object).value(a),instance.value(a)), value);
						if (value==0) break;
					}
				}
				//value*=(m_Train.numAttributes()-rule.attributes.cardinality());

				distribution[(int)m_Train.instance(rule.object).classValue()]+=value;
				//classCounts[(int)m_Train.instance(rule.object).classValue()]++;
				total+=value;

				/*if (value>bestValue) {
					bestValue=value;
					bestClass = (int)m_Train.instance(rule.object).value(m_classIndex);
				}
				if (bestValue==1) break;*/
			}
			//if (bestClass>-1) distribution[bestClass]=1;

			//for (int i=0;i<m_numClasses;i++) System.err.print(classCounts[i]+" ");
			//System.err.println();

			if (total>0) Utils.normalize(distribution, total);
		}
		else {//if (m_ClassType == Attribute.NUMERIC) {
			double denom=0; double num=0;

			Iterator<FuzzyRule> it = cands.iterator();
			FuzzyRule rule;

			while (it.hasNext()) {
				rule = it.next();
				double value=1;

				for (int a = rule.attributes.nextSetBit(0); a >= 0; a = rule.attributes.nextSetBit(a + 1)) {
					if (a != m_classIndex) {
						value = m_composition.calculate(fuzzySimilarity(a,m_Train.instance(rule.object).value(a),instance.value(a)), value);
						if (value==0) break;
					}
				}

				num+=value*m_Train.instance(rule.object).value(m_classIndex);
				denom+=value;
			}
			distribution[0] = num/denom;
		}
		
		return distribution;
	}



	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain command line options (see setOptions)
	 */
	public static void main(String [] argv) {
		runClassifier(new EntropyRules(), argv);
	}
}
