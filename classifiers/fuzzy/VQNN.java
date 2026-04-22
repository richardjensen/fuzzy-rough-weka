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


import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.FuzzyRoughAttributeEval;
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

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;
import java.util.Collections;


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
public class VQNN 
extends FuzzyRoughClassifier 
implements OptionHandler, TechnicalInformationHandler {

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;
	
	//parameters for the fuzzy quantifier
	public double alpha_u=0.2;
	public double beta_u=1.0;
	public double alpha_l=0.1;
	public double beta_l=0.6;


	/**
	 * Fuzzy rough IBk classifier. Simple instance-based learner that uses the class
	 * of the nearest k training instances for the class of the test
	 * instances.
	 *
	 * @param k the number of nearest neighbors to use for prediction
	 */
	public VQNN(int k) {
		init();
		setKNN(k);
	}  

	
	public VQNN() {
		init();
	}

	/**
	 * Returns a string describing classifier.
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return  "Vaguely-quantified K-nearest neighbours classifier.\n\n"
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

		String knnString = Utils.getOption('K', options);
		if (knnString.length() != 0) {
			setKNN(Integer.parseInt(knnString));
		} else {
			setKNN(1);
		}
		
		setUseFeatureWeighting(Utils.getFlag('F', options));
		
		knnString = Utils.getOption('A', options);
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
		
		knnString = Utils.getOption('C', options);
		if (knnString.length() != 0) {
			setAlphaL(Double.valueOf(knnString));
		} else {
			setAlphaL(0.1);
		}
		
		knnString = Utils.getOption('D', options);
		if (knnString.length() != 0) {
			setBetaL(Double.valueOf(knnString));
		} else {
			setBetaL(0.6);
		}

		String optionString  = Utils.getOption('T', options);
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

		optionString = Utils.getOption('X', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid ASEvaluation specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setRanker( (ASEvaluation) Utils.forName( ASEvaluation.class, className, moreOptions) );
		}
		else {
			setRanker(new FuzzyRoughAttributeEval());
		}
		
		Utils.checkForRemainingOptions(options);
	}

	public void setBetaL(double valueOf) {
		beta_l=valueOf;
	}


	public void setAlphaL(double valueOf) {
		alpha_l=valueOf;
	}


	public void setBetaU(double valueOf) {
		beta_u=valueOf;
	}


	public void setAlphaU(double valueOf) {
		alpha_u=valueOf;
	}

	public double getBetaL() {
		return beta_l;
	}
	
	public double getAlphaL() {
		return alpha_l;
	}
	public double getBetaU() {
		return beta_u;
	}
	public double getAlphaU() {
		return alpha_u;
	}


	public void setTNorm(TNorm tnorm) {
		m_TNorm = tnorm;
	}

	public TNorm getTNorm() {
		return m_TNorm;
	}
	
	
	/**
	 * Gets the current settings of IBk.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String [] getOptions() {

		String [] options = new String [18];
		int current = 0;
		options[current++] = "-K"; options[current++] = "" + getKNN();
		if (getUseFeatureWeighting()) options[current++] = "-F";
		
		options[current++] = "-A"; options[current++] = "" + getAlphaU();
		options[current++] = "-B"; options[current++] = "" + getBetaU();
		options[current++] = "-C"; options[current++] = "" + getAlphaL();
		options[current++] = "-D"; options[current++] = "" + getBetaL();
		
		
		options[current++] = "-T";
		options[current++] = (m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim();
		
		options[current++] = "-R";
		options[current++] = (m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim(); 

		options[current++] = "-X";
		options[current++] = (ranker.getClass().getName() + " ").trim(); 
		
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
			return "VQNN: No model built yet.";
		}

		String result = "VQNN\n" +
		"using " + m_kNN;
		result += " nearest neighbour(s) for classification\n\n";
		result += "Using parameters alpha_u="+alpha_u+", beta_u="+beta_u+", alpha_l="+alpha_l+", beta_l="+beta_l+"\n";
		StringBuffer text = new StringBuffer();
		text.append("\nSimilarity measure: "+m_Similarity);
		text.append("\nT-Norm: "+m_TNorm+"\nRelation composition: "+m_Similarity.getTNorm());
		result+=text.toString();
		
		return result;
	}

	/**
	 * Initialise scheme variables.
	 */
	protected void init() {
		setKNN(10);
		//m_composition = new TNormKD();
		
	    m_TNorm = new TNormKD();
	    m_Similarity = new Similarity1();
	    
		try {
			m_Similarity.setTNorm(new TNormAlgebraic());
			//m_NNSearch.setDistanceFunction(new Similarity1());
			//(m_NNSearch.getDistanceFunction()).getTNorm();
			
		}
		catch (Exception e) {}
	}


	public TNorm m_TNorm = new TNormKD();
	
	public Instances kNearestNeighbours(Instance instance, int m_kNN) {
		ArrayList<Index> objectIndex = new ArrayList<Index>(m_Train.numInstances());
		
		for(int o=0; o<m_Train.numInstances(); o++) {
			double sim=1;
			
			sim=0;
			
			//compute similarity
			for(int i = 0; i< m_Train.numAttributes(); i++){
				if (i != m_Train.classIndex()) {
					double sim1 = fuzzySimilarity(i, instance.value(i), m_Train.instance(o) .value(i));
					
					//sim = m_composition.calculate(sim,sim1);
					sim += weightings[i]*sim1*sim1;
				}
			}
			sim /=m_Train.numAttributes();
			
			objectIndex.add(new Index(o, sim));
		}
		Collections.sort(objectIndex);
		Instances ret = new Instances(m_Train, m_kNN);
		
		for (int o=0;o<m_kNN;o++) {
			Instance ins = m_Train.instance(objectIndex.get(o).object);
			ins.setWeight(objectIndex.get(o).sim);
			ret.add(ins);
			
		}
		
		return ret;
	}
	
	/**
	 * Turn the list of nearest neighbours into a probability distribution.
	 *
	 * @param neighbours the list of nearest neighbouring instances
	 * @param distances the distance of the neighbours
	 * @return the probability distribution
	 * @throws Exception if computation goes wrong or has no class attribute
	 */
	protected double [] makeDistribution(Instances neighbours)
	throws Exception {

		double [] distribution=null;
		double lower;
		double upper;
		double bestLower=-1;
		double bestUpper=-1;
		double total=0;

		//for normalization
		double norm=neighbours.instance(0).weight();	
		if (norm==0) norm=1;
		
		int classIndex = neighbours.classIndex();
		
		// If we're performing imputation for missing values...
		int attrToPredict = getAltClassIndex();
		if (attrToPredict>=0) {
			classIndex = attrToPredict;
			m_ClassType = neighbours.instance(0).attribute(attrToPredict).type();
			m_NumClasses = neighbours.instance(0).attribute(attrToPredict).numValues();
		}
		
		
		if (m_ClassType == Attribute.NOMINAL) {
			distribution = new double [m_NumClasses];

			for(int i = 0; i < m_NumClasses; i++) {
				double val = 0;
				double denom = 0;
				
				//work out upper and lower approximations
				for(int o=0; o < neighbours.numInstances(); o++) {
					Instance neighbour = neighbours.instance(o);
					double dec =0;
					
					//if ((int)neighbour.classValue() == i) {dec=1;}
					if ((int)neighbour.value(classIndex) == i) {dec=1;}
										
					val+=m_TNorm.calculate(neighbour.weight()/norm, dec);
					denom+= (neighbour.weight()/norm);
					
				}

				lower = Q(val / denom, alpha_u, beta_u);
				upper = Q(val / denom, alpha_l, beta_l);
			
				if (lower>=bestLower) {
					if (upper>=bestUpper) {
						bestLower=lower;
						bestUpper=upper;
					}
				}
				
				distribution[i] = (upper+lower)/2;
				total+=distribution[i];

			}
			if (total>0) Utils.normalize(distribution, total);
		}
		else {//if (m_ClassType == Attribute.NUMERIC) {
			//int classIndex = m_Train.classIndex();
			double denom=0;double num=0;
			
			for(int i = 0; i < neighbours.numInstances(); i++) {
				Instance neighbour = neighbours.instance(i);
				double val1 = 0;
				double denom1 = 0;
				
				//work out VQ upper and lower approximations
				for(int o=0; o < neighbours.numInstances(); o++) {
					Instance current = neighbours.instance(o);
					
					//double dec = m_Similarity.similarity(classIndex, neighbour.value(classIndex), current.value(classIndex));
					double dec = fuzzySimilarity(classIndex, neighbour.value(classIndex), current.value(classIndex));
					
					val1+=m_TNorm.calculate(current.weight()/norm, dec);
					denom1+=current.weight()/norm;
					
				}
				lower = Q(val1 / denom1, alpha_u, beta_u);
				upper = Q(val1 / denom1, alpha_l, beta_l);
				
				
				double measure = (lower+upper)/2;
				if (!neighbour.isMissing(classIndex)) {
					num+= (measure*neighbour.value(classIndex));
					denom+=measure;
				}
			}
			
			distribution = new double[neighbours.numInstances()];
			if (denom==0) distribution[0] =  neighbours.instance(0).value(classIndex);
			else distribution[0] = num/denom;
		}	
		
		return distribution;
	}
	
	//fuzzy quantifier
	protected final double Q(double x, double alpha, double beta) {
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
		runClassifier(new VQNN(), argv);
	}
}
