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
import weka.fuzzy.tnorm.*;
import weka.fuzzy.similarity.*;

import java.util.*;


/**
 <!-- globalinfo-start -->
 * Vaguely-quantified fuzzy-rough nearest-neighbour classifier. <br/>
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
public class VQRules 
extends FuzzyRoughRuleInducer 
implements OptionHandler,
TechnicalInformationHandler {

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;

	//parameters for the fuzzy quantifier
	public double alpha_u=0.2;
	public double beta_u=1.0;
	public double alpha_l=0.1;
	public double beta_l=0.6;
	
	//whether to use the weak positive region or not
	public boolean m_weak=true;
	
	public VQRules() {
		init();
	}

	/**
	 * Returns a string describing classifier.
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return  "Vaguely-quantified fuzzy-rough rule induction.\n\n"
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
				"\tNumber of nearest neighbours (k) used in classification.\n"+
				"\t(Default = 1)",
				"K", 1,"-K <number of neighbors>"));
		newVector.addElement(new Option(
				"\tSelect the number of nearest neighbours between 1\n"+
				"\tand the k value specified using hold-one-out evaluation\n"+
				"\ton the training data (use when k > 1)",
				"X", 0,"-X"));
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

		String optionString;


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

		setWeak(Utils.getFlag('W', options));
		
		String knnString = Utils.getOption('A', options);
		if (knnString.length() != 0) {
			setAlphaU(Double.valueOf(knnString));
		} else {
			setAlphaU(0.2);
		}
		
		knnString = Utils.getOption('B', options);
		if (knnString.length() != 0) {
			setBetaU(Double.valueOf(knnString));
		} else {
			setBetaU(1);
		}
		
		Utils.checkForRemainingOptions(options);
	}

	public void setBetaU(double valueOf) {
		beta_u=valueOf;
	}


	public void setAlphaU(double valueOf) {
		alpha_u=valueOf;
	}

	public double getBetaU() {
		return beta_u;
	}
	public double getAlphaU() {
		return alpha_u;
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
		
		options[current++] = "-A"; options[current++] = "" + getAlphaU();
		options[current++] = "-B"; options[current++] = "" + getBetaU();
		
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
		//m_DecisionSimilarity.setInstances(m_Train);
		//m_SimilarityEq.setInstances(m_Train);

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

		if (m_ClassType != Attribute.NOMINAL&&!m_weak) fullGamma = calculateGamma2(current,full);
		else fullGamma = calculateGamma(current,full);

		System.err.println(fullGamma);
		
		fullData=false;
		if (m_Pruning<0||m_Pruning>10) m_Pruning=0;
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
					if (m_ClassType != Attribute.NOMINAL&&!m_weak) gam = calculateGamma2(temprel,reduct);
					else gam = calculateGamma(temprel,reduct);

					System.err.println(reduct+" => "+gam);
					reduct.clear(a);

					if (gam>=gamma) {
						gamma=gam;
						bestAttr=a;
					}
				}
			}

			if (bestAttr==-1)  break;

			reduct.set(bestAttr);
			
			if (m_Debug) System.err.println(reduct+" => "+gamma+" "+full);
			if (allCovered()) break;
			
			generatePartition(current,bestAttr);
			setCurrent(temprel);			
		}
		
		for (int i=0;i<m_Pruning;i++) prune();
		
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


	//weak version
	private final double calculateGamma(Relation rel, BitSet reduct) {
		double lower;
		double ret=0;
		int d=-1;
		double[] eqClass;

		for (int x=0;x<m_numInstances;x++) {
			d=x;//weak version, so only consider x's decision
			lower=1;

			eqClass = new double[m_numInstances];
			double sum=0;
			double denom=0;
			
			//lower  approximations of object x
			for (int y=0;y<m_numInstances;y++) {
				double val=1;

				val = rel.getCell(x,y);

				eqClass[y] = val;

				sum += m_TNorm.calculate(val,fuzzySimilarity(m_classIndex, m_Train.instance(d).value(m_classIndex), m_Train.instance(y).value(m_classIndex)));
				denom+=val;

			}
			lower = Q(sum / denom, alpha_u, beta_u);
			
			if ((float)lower==(float)globallowers[x] && global[x]<globallowers[x]) {
				determineAdd(new FuzzyRule(eqClass,(BitSet)reduct.clone(),x));
			}
			if (fullData) globallowers[x]=lower;
			ret+=lower;

		}
		
		return Math.min(1, ret/Double.valueOf(m_numInstances));
	}

	//full version
	private final double calculateGamma2(Relation rel, BitSet reduct) {
		double lower;
		double ret=0;
		double[] eqClass; double val;
		vals = new double[m_numInstances];
		indexes = new int[m_numInstances];

		setIndexes(rel);

		for (int x = 0; x < m_numInstances; x++) {
			if (indexes[x] != 0) {
				vals[x] = vals[indexes[x] - 1];
			} else {
				val = 0;
				decVals = new double[m_numInstances];
				
				
				for (int d = 0; d < m_numInstances; d++) {
					if (decIndexes[d] != 0) {
						decVals[d] = decVals[decIndexes[d] - 1];
					} else {
						lower = 1;
						eqClass = new double[m_numInstances];
						double sum=0;
						double denom=0;
						
						// lower approximations of object x
						for (int y = 0; y < m_numInstances; y++) {
							double condSim = rel.getCell(x, y);
							eqClass[y] = condSim;
							sum += m_TNorm.calculate(condSim, fuzzySimilarity(m_classIndex, m_Train.instance(d).value(m_classIndex), m_Train.instance(y).value(m_classIndex)));
							denom+=condSim;
						}
						
						lower = Q(sum / denom, alpha_u, beta_u);
						decVals[d] = lower;
						
						//if (decVals[d]==globallowers[x][d]&&global[x]/globalpositives[x]<1) {
						if (!fullData&&(float)decVals[d]==(float)globallowers[x]&&global[x]<globallowers[x]) {
							determineAdd(new FuzzyRule(eqClass,(BitSet)reduct.clone(),x));
						}
					}
					//if (fullData) globallowers[x][d]=decVals[d];
					val = Math.max(decVals[d], val);
					
					
					if (val == 1) break;
				}

				vals[x] = val;
				
			}
			if (fullData) globallowers[x]=vals[x];
			ret += vals[x];
		}


		return Math.min(1, ret/Double.valueOf(m_numInstances));
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
//		double bestValue=-1;
//		int bestClass=-1;

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
						if (value==0) break;
					}
				}
				//value*=(m_Train.numAttributes()-rule.attributes.cardinality());

				distribution[(int)m_Train.instance(rule.object).classValue()]+=value;
				total+=value;
			}

			if (total>0) Utils.normalize(distribution, total);
		}
		else {//if (m_ClassType == Attribute.NUMERIC) {
			double denom=0;double num=0;

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

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain command line options (see setOptions)
	 */
	public static void main(String [] argv) {
		runClassifier(new VQRules(), argv);
	}
}
