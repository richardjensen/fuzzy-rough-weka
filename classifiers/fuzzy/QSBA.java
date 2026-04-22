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
import java.io.Serializable;

import java.util.*;


/**
 <!-- globalinfo-start -->
 * QSBA. <br/>
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
public class QSBA 
extends FuzzyRoughRuleInducer 
implements OptionHandler,
TechnicalInformationHandler {

	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;

	//whether to use the weak positive region or not
	public boolean m_weak=true;

	public class Hood implements Comparable<Hood> {
		String term;
		double subsethood;
		int index=0;//index of linguistic term in parent attribute array
		int setIndex=-1; //which parent fuzzy set it belongs to

		public Hood(String t, double s, int i, int si) {
			term = t;
			subsethood=s;
			index = i;
			setIndex = si;
		}

		public int compareTo(Hood h) {
			return ((int)(100000*h.subsethood)-(int)(100000*subsethood));
		}

		public String toString() {
			return term+" "+subsethood;
		}
	}

	//part of a rule, e.g. A is (0.5*A1 OR 0.2*A2 OR 1.0*A3)
//	also deals with the default rule case
	public final class Part implements Serializable {
		static final long serialVersionUID = -3080186098777067171L;
		int attr;
		int attr_index;
		boolean decision;// is this part the decision?
		boolean defaultRule; // is this a default rule?
		double[] weight; //the weights for the terms
		double beta;
		double dec;

		public Part(int a, int ai, int weightSize) {
			attr=a;
			attr_index=ai;
			weight = new double[weightSize];
			defaultRule=false;
		}

		public Part(double b) {
			decision=true;
			dec=b;
		}

		public void setWeight(int i, double val) {
			weight[i] = val;
		}

		public double getWeight(int i) {
			return weight[i];
		}

		public String toString() {
			String ret="";

			if (decision) ret= " THEN "+attr+" IS "+dec;
			else {
				ret=attr+" IS (";
				for (int a=0;a<weight.length;a++) {
					ret+=weight[a]+"*"+a;
					if (a==weight.length-1) break;
					else ret+= " OR ";
				}
				ret+=") ";
			}
			return ret;
		}
	}

	public QSBA() {
		init();
	}

	/**
	 * Returns a string describing classifier.
	 * @return a description suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return  "QSBA.\n\n"
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


		Utils.checkForRemainingOptions(options);
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

		String [] options = new String [15];
		int current = 0;



		options[current++] = "-T";
		options[current++] = (m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim();



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


		text.append("\nT-Norm: "+m_TNorm);
		text.append("\n\nNumber of rules: "+rules.length);
		text.append("\nAverage rule arity: "+arity);

		result+=text.toString();

		return result;
	}

	/**
	 * Initialise scheme variables.
	 */
	protected void init() {
		m_TNorm = new TNormAlgebraic();
		m_SNorm = m_TNorm.getAssociatedSNorm();
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
		
		m_SNorm = m_TNorm.getAssociatedSNorm();
		
		m_numClasses = instances.numClasses();
		m_ClassType = instances.classAttribute().type();
		m_Train = new Instances(instances, 0, instances.numInstances());
		m_numAttributes = m_Train.numAttributes();

		m_classIndex = m_Train.classIndex();
		m_numInstances = m_Train.numInstances();

		arity=0;
		setUpFuzzySets();
		induce(-1);
	}


	FuzzySet[][] sets;
	//5 fuzzy sets
	public int numSets=5;
	public int m_numAttributes;
	public double threshold=0.7;
	double universal=0;
	double existential=0;
	ArrayList<Part>[] rules; 
	BitSet[] subgroup;

	/**
	 * Induce (and store) rules via QuickRules
	 */
	public void induce(double full) {
		subgroup = new BitSet[m_numClasses];
		rules = new ArrayList[m_numClasses];

		//set up subgroup
		for (int i=0;i<subgroup.length;i++) {
			subgroup[i] = new BitSet(m_numInstances);
		}
		for (int o=0;o<m_numInstances;o++) {
			subgroup[(int)m_Train.instance(o).classValue()].set(o);
		}

		//for each subgroup/plan
		for (int i=0;i<subgroup.length;i++) {

			BitSet Plan = subgroup[i];
			rules[i] = new ArrayList<Part>(m_numAttributes);

			//for each attribute's linguistic terms, work out subsethood
			//with corresponding plan
			for (int attr=0;attr<m_numAttributes-1;attr++) {
				if (attr!=m_classIndex&&!m_Train.attribute(attr).isNominal()) {
					Part part = new Part(attr,-1,sets[attr].length);

					for (int index=0;index<sets[attr].length;index++) {
						FuzzySet LT = sets[attr][index];
						double lambda = subsethood(Plan,LT,i,attr);
						Q(Plan,LT,i,attr); //work out the quantifiers

						double finalQuantifier = (1-lambda)*universal + (lambda)*existential;
						//System.err.print(finalQuantifier+"*"+index);

						part.setWeight(index,finalQuantifier);

					}
					rules[i].add(part);



					//final bit of rule information for this particular Plan
					int obj;
					for (obj=0;obj<m_numInstances;obj++) if (Plan.get(obj)) break;

					Part p = new Part(m_Train.instance(obj).value(m_classIndex));
					p.attr=m_classIndex;

					rules[i].add(p);
				}
			}
		}

	}


	private final double subsethood(BitSet A, FuzzySet B, int subIndex, int att) {
		double sum=0;
		double current=0;
		double Amem;
		double Bmem;

		for (int o=0;o<m_numInstances;o++) {
			if (subgroup[subIndex].get(o)) {
				if(A.get(o)) Amem=1;
				else Amem=0;
				Bmem = B.getMembership(m_Train.instance(o).value(att));
				sum+=Amem;
				current+=m_TNorm.calculate(Amem,Bmem);
			}
		}
		return current/sum; //equation 5
	}


	//this updates the existential and universal values at the same time
	private final void Q(BitSet A, FuzzySet B, int subIndex,int att) {
		double Amem;
		double Bmem;
		existential=0;
		universal = 1;//this is 1 as we're t-norming

		for (int o=0;o<m_numInstances;o++) {
			if (subgroup[subIndex].get(o)) {
				if(A.get(o)) Amem=1; //decision membership
				else Amem=0;
				Bmem = B.getMembership(m_Train.instance(o).value(att));
				existential = m_SNorm.calculate(m_TNorm.calculate(Bmem,Amem),existential);
				universal = m_TNorm.calculate(m_SNorm.calculate(Bmem,1-Amem),universal);
			}
		}
	}

	//constructs the fuzzy sets for each attribute
	public void setUpFuzzySets() {
		sets = new FuzzySet[m_numAttributes][numSets];

		for (int a=0;a<m_numAttributes;a++) {
			if (a!=m_classIndex) {
				double std;
				if (m_Train.attribute(a).isNumeric()) std = Math.sqrt(m_Train.variance(a));
				else std = (m_Train.attribute(a).getUpperNumericBound() - m_Train.attribute(a).getLowerNumericBound())/2;
				
				double mean = m_Train.meanOrMode(a);

				double point1 = mean - threshold*std;
				double point2 = mean + threshold*std;

				sets[a][0] = new FuzzySet("Low",a,(mean-2*std),(mean-std),-1,-1,FuzzySet.L_SHOULDERED);
				sets[a][1] = new FuzzySet("MedL",a,(mean-2*std),(mean-std),point1,-1,FuzzySet.TRIANGULAR);
				sets[a][2] = new FuzzySet("Medium",a,(mean-std),point1,point2,(mean+std),FuzzySet.TRAPEZOIDAL);
				sets[a][3] = new FuzzySet("MedH",a,point2,(mean+std),(mean+2*std),-1,FuzzySet.TRIANGULAR);
				sets[a][4] = new FuzzySet("High",a,(mean+std),(mean+2*std),-1,-1,FuzzySet.R_SHOULDERED);
			}
		}
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

		if (m_ClassType == Attribute.NOMINAL) {
			distribution = new double [m_numClasses];

			for (int rule=0;rule<m_numClasses;rule++) {
				distribution[rule]=makeClassification(instance,rules[rule]);
				total+=distribution[rule];
			}



			if (total>0) Utils.normalize(distribution, total);
		}
		else {//if (m_ClassType == Attribute.NUMERIC) {

		}

		return distribution;
	}

	private final double makeClassification(Instance obj, ArrayList<Part> rule) {
		double ret=1;

		//each rule is made up of one Part per attribute, with the final
		//Part being the decision
		Iterator<Part> it = rule.iterator();
		double mem=0;
		double oldmem=0;

		//for each attr...
		while(it.hasNext()) {
			Part p = it.next();
			mem=0;
			if (!p.decision) {//not a decision
				//for each attribute's linguistic term work out
				//the highest weighted membership
				for (int s=0;s<p.weight.length;s++) {
					oldmem = ((FuzzySet)sets[p.attr][s]).getMembership(obj.value(p.attr));
					oldmem = m_TNorm.calculate(oldmem,p.weight[s]);

					mem = m_SNorm.calculate(mem,oldmem);
				}
				ret = m_TNorm.calculate(ret,mem);

				if (ret==0) break;
			}

		}
		return ret;
	}

	/**
	 * Main method for testing this class.
	 *
	 * @param argv should contain command line options (see setOptions)
	 */
	public static void main(String [] argv) {
		runClassifier(new QSBA(), argv);
	}
}
