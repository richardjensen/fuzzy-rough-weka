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
 *    FuzzyNN.java
 *
 */


import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
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
import weka.fuzzy.similarity.*;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;
import java.util.Collections;


/**
 <!-- globalinfo-start -->
 * Fuzzy nearest-neighbour classifier. <br/>
 * <br/>
 * For more information, see <br/>
 * <br/>
 * D. Aha, D. Kibler (1991). Instance-based learning algorithms. Machine Learning. 6:37-66.
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
public class FuzzyNNold 
extends FuzzyRoughClassifier 
implements OptionHandler, TechnicalInformationHandler {

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;
	
	/**
	 * Fuzzy rough IBk classifier. Simple instance-based learner that uses the class
	 * of the nearest k training instances for the class of the test
	 * instances.
	 *
	 * @param k the number of nearest neighbors to use for prediction
	 */
	public FuzzyNNold(int k) {

		init();
		setKNN(k);
	}  

	
	public FuzzyNNold() {

		init();
	}

	/**
	 * Returns a string describing classifier.
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return  "Fuzzy K-nearest neighbours classifier.\n\n"
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

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "D. Aha and D. Kibler");
		result.setValue(Field.YEAR, "1991");
		result.setValue(Field.TITLE, "Instance-based learning algorithms");
		result.setValue(Field.JOURNAL, "Machine Learning");
		result.setValue(Field.VOLUME, "6");
		result.setValue(Field.PAGES, "37-66");

		return result;
	}


	public void setFuzzifier(double d) {
		m_fuzzifier=d;
	}
	
	public double getFuzzifier() {
		return m_fuzzifier;
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

		return newVector.elements();
	}

	/**
	 * Parses a given list of options. <p/>
	 *
 <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -K &lt;number of neighbors&gt;
	 *  Number of nearest neighbours (k) used in classification.
	 *  (Default = 1)</pre>
	 * 
	 * 
 <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {

		String optionString = Utils.getOption('K', options);
		if (optionString.length() != 0) {
			setKNN(Integer.parseInt(optionString));
		} else {
			setKNN(10);
		}
		
		optionString = Utils.getOption('Q', options);
		if (optionString.length() != 0) {
			setFuzzifier(Double.valueOf(optionString));
		} else {
			setFuzzifier(3);
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
			setSimilarity(new SimilarityFNN());
		}

		Utils.checkForRemainingOptions(options);
	}

	
	/**
	 * Gets the current settings of IBk.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String [] getOptions() {

		String [] options = new String [15];
		int current = 0;
		options[current++] = "-K"; options[current++] = "" + getKNN();

		options[current++] = "-Q"; options[current++] = "" + getFuzzifier();
		
		options[current++] = "-R";
		options[current++] = (m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim(); 

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
			return "FuzzyNN: No model built yet.";
		}

		String result = "FuzzyNN\n" +
		"using " + m_kNN;
		result += " nearest neighbour(s) for classification\n";


		return result;
	}

	/**
	 * Initialise scheme variables.
	 */
	protected void init() {
		setKNN(10);		
	    m_Similarity = new SimilarityFNN();
	    
	}
	

	//weighting
	public double m_fuzzifier=3;
	
	public Instances kNearestNeighbours(Instance instance, int m_kNN) {
		ArrayList<Index> objectIndex = new ArrayList<Index>(m_Train.numInstances());
		
		for(int o=0; o<m_Train.numInstances(); o++) {
			double sim=1;
			
			//compute similarity
			for(int i = 0; i< m_Train.numAttributes(); i++){
				if (i != m_Train.classIndex()) {
					double mainVal = instance.value(i); 
					double otherVal = m_Train.instance(o) .value(i);
						
					sim += fuzzySimilarity(i,mainVal,otherVal);
				}
			}
			
			sim = 1 / Math.pow(sim, 1 / (m_fuzzifier - 1));
			
			objectIndex.add(new Index(o, sim));
			m_Train.instance(o).setWeight(sim);
		}
		Collections.sort(objectIndex);
		Instances ret = new Instances(m_Train, m_kNN);
		
		for (int o=0;o<m_kNN;o++) {
			ret.add(m_Train.instance(objectIndex.get(o).object));
			
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
	protected double [] makeDistribution(Instances neighbours) throws Exception {
		double [] distribution = new double [m_NumClasses];
		double classification = -1;
		double sum = 0;
		double denom = 0;
		double best=-1;
		int bestClass=-1;
		
		if (m_ClassType == Attribute.NOMINAL) {
			
			for(int i = 0; i < m_NumClasses; i++) {
				classification = -1;
				sum = 0;
				denom = 0;
				
				for(int o=0; o < neighbours.numInstances(); o++) {
					Instance current = neighbours.instance(o);
					double dec =0;
					
					if ((int)current.classValue() == i) {dec=1;}
					
					double val = current.weight();
					
					sum+=val*dec;
					denom+=val;
				}
				
				if (denom==0) classification=0;
				else classification = sum/denom;
				
				if (classification>best) {
						best=classification;
						bestClass=i;	
				}
				distribution[i]=0;

			}
			distribution[bestClass]=1;
		}
		else {
			int classIndex = m_Train.classIndex();
			double num=0;
			
			for(int i = 0; i < neighbours.numInstances(); i++) {
				Instance object = neighbours.instance(i);
				
				for(int o=0; o < neighbours.numInstances(); o++) {
					Instance current = neighbours.instance(o);
					
					double val = current.weight();
					
					num+=val*object.value(classIndex);
					denom+=val;
					
					
					
				}
			}
			
			distribution = new double[neighbours.numInstances()];
			distribution[0] = num/denom;
		}


		// Normalise distribution
		//if (total>0) Utils.normalize(distribution, total);
		
		return distribution;
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain command line options (see setOptions)
	 */
	public static void main(String [] argv) {
		runClassifier(new FuzzyNNold(), argv);
	}
}
