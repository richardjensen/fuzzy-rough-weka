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
import weka.fuzzy.implicator.*;
import weka.fuzzy.tnorm.*;
import weka.fuzzy.similarity.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;


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
public class FuzzyRoughNN 
extends FuzzyRoughClassifier 
implements OptionHandler, TechnicalInformationHandler {

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;

	/**
	 * Fuzzy rough NN classifier. Simple instance-based learner that uses the class
	 * of the nearest k training instances for the class of the test
	 * instances.
	 *
	 * @param k the number of nearest neighbours to use for prediction
	 */
	public FuzzyRoughNN(int k) {
		init();
		setKNN(k);
	}  


	public FuzzyRoughNN() {
		init();
	}

	/**
	 * Returns a string describing classifier.
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return  "Fuzzy-rough K-nearest neighbours classifier.\n\n"
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
		//result.enable(Capability.MISSING_CLASS_VALUES);

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
	 * <pre> -I
	 *  Weight neighbours by the inverse of their distance
	 *  (use when k &gt; 1)</pre>
	 * 
	 * <pre> -F
	 *  Weight neighbours by 1 - their distance
	 *  (use when k &gt; 1)</pre>
	 * 
	 * <pre> -K &lt;number of neighbors&gt;
	 *  Number of nearest neighbours (k) used in classification.
	 *  (Default = 1)</pre>
	 * 
	 * <pre> -E
	 *  Minimise mean squared error rather than mean absolute
	 *  error when using -X option with numeric prediction.</pre>
	 * 
	 * <pre> -W &lt;window size&gt;
	 *  Maximum number of training instances maintained.
	 *  Training instances are dropped FIFO. (Default = no window)</pre>
	 * 
	 * <pre> -X
	 *  Select the number of nearest neighbours between 1
	 *  and the k value specified using hold-one-out evaluation
	 *  on the training data (use when k &gt; 1)</pre>
	 * 
	 * <pre> -A
	 *  The nearest neighbour search algorithm to use (default: weka.core.neighboursearch.LinearNNSearch).
	 * </pre>
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

		String optionString = Utils.getOption('I', options);
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

		// for use in imputation - set another attribute as the 'class' attribute
		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setAltClassIndex(Integer.valueOf(optionString));
		} else {
			setAltClassIndex(-1);
		}


		Utils.checkForRemainingOptions(options);
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

		String [] options = new String [20];
		int current = 0;
		options[current++] = "-K"; options[current++] = "" + getKNN();

		if (getUseFeatureWeighting()) options[current++] = "-F";

		options[current++] = "-I";
		options[current++] = (m_Implicator.getClass().getName() + " " +
				Utils.joinOptions(m_Implicator.getOptions())).trim();

		options[current++] = "-T";
		options[current++] = (m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim();

		options[current++] = "-R";
		options[current++] = (m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim(); 

		options[current++] = "-X";
		options[current++] = (ranker.getClass().getName() + " ").trim(); 

		options[current++] = "-A";
		options[current++] = String.valueOf(getAltClassIndex());

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
			return "FuzzyRough NN: No model built yet.";
		}

		String result = "FuzzyRough NN\n" +
				"using " + m_kNN;
		result += " nearest neighbour(s) for classification\n";
		StringBuffer text = new StringBuffer();
		text.append("\nSimilarity measure: "+m_Similarity);
		text.append("\nImplicator: "+m_Implicator);
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
		m_Implicator = new ImplicatorKD();
		m_Similarity = new Similarity1();

		try {
			m_Similarity.setTNorm(new TNormAlgebraic());
			//m_NNSearch.setDistanceFunction(new Similarity1());
			//(m_NNSearch.getDistanceFunction()).getTNorm();

		}
		catch (Exception e) {}
	}

	public TNorm m_TNorm = new TNormKD();
	public Implicator m_Implicator = new ImplicatorKD();

	public Instances kNearestNeighbours(Instance instance, int m_kNN) {
		ArrayList<Index> objectIndex = new ArrayList<Index>(m_Train.numInstances());

		for(int o=0; o<m_Train.numInstances(); o++) {
			double sim=1;

			sim=0;

			//compute similarity
			for(int i = 0; i< m_Train.numAttributes(); i++){
				if (i != m_Train.classIndex()) {
					double sim1 = fuzzySimilarity(i, instance.value(i), m_Train.instance(o).value(i));

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
	protected double[] makeDistribution(Instances neighbours)
			throws Exception {
		double[] distribution=null;

		double total=0;
		double lower = 1;
		double upper = 0;

		//for normalization
		double norm = neighbours.instance(0).weight();
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
				lower = 1;
				upper = 0;

				//work out upper and lower approximations
				for(int o=0; o < neighbours.numInstances(); o++) {
					Instance neighbour = neighbours.instance(o);
					double dec =0;

					// original if ((int)neighbour.classValue() == i) {dec=1;}
					if ((int)neighbour.value(classIndex) == i) {dec=1;} //check that this works!

					double val = neighbour.weight()/norm;
					double impl = m_Implicator.calculate(val, dec);
					double tnorm = m_TNorm.calculate(val, dec);

					lower = Math.min(impl, lower);
					upper = Math.max(tnorm, upper);
				}

				distribution[i] = (upper+lower)/2;
				total+=distribution[i];

			}

			if (total>0) Utils.normalize(distribution, total);
		}
		else {//if (m_ClassType == Attribute.NUMERIC) {
			double denom=0;
			double num=0;

			for (int i = 0; i < neighbours.numInstances(); i++) {
				Instance neighbour = neighbours.instance(i);
				lower = 1;
				upper = 0;

				//work out upper and lower approximations
				for (int o=0; o < neighbours.numInstances(); o++) {
					Instance current = neighbours.instance(o);

					double dec = fuzzySimilarity(classIndex, neighbour.value(classIndex), current.value(classIndex));

					lower = Math.min(m_Implicator.calculate(current.weight()/norm, dec),lower);
					upper = Math.max(m_TNorm.calculate(current.weight()/norm, dec), upper);

				}

				double measure = (lower+upper)/2;
				if (!neighbour.isMissing(classIndex)) {
					num+= (measure*neighbour.value(classIndex));
					denom+=measure;
				}
			}
			//System.out.println(num);

			distribution = new double[neighbours.numInstances()];
			if (denom==0) distribution[0] =  neighbours.instance(0).value(classIndex);
			else distribution[0] = num/denom;
		}



		return distribution;
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain command line options (see setOptions)
	 */
	public static void main(String [] argv) {
		runClassifier(new FuzzyRoughNN(), argv);
	}
}
