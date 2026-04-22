package weka.attributeSelection;


import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.measure.*;
import weka.fuzzy.ivmeasure.*;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity3;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormLukasiewicz;

import java.util.BitSet;
import java.util.Enumeration;
import java.util.Vector;



public class FuzzyRoughAttributeEval 
extends ASEvaluation
implements AttributeEvaluator, OptionHandler, TechnicalInformationHandler { 

	/** for serialization */
	static final long serialVersionUID = 747878400839276317L;

	/** The training instances */
	private Instances m_trainInstances;
	/** The class index */
	private int m_classIndex;

	/** Number of attributes in the training data */
	private int m_numAttribs;
	/** Number of instances in the training data */
	private int m_numInstances;
	private boolean m_isNumeric;

	// normalising factor
	protected double c_divisor = -1; // may need to change this

	public Similarity m_Similarity = new Similarity3();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity = new SimilarityEq();
	protected Measure m_Measure= new WeakGamma();


	public TNorm m_TNorm = new TNormLukasiewicz();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();

	/**
	 * Returns a string describing this attribute evaluator
	 * @return a description of the evaluator suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Rough and fuzzy rough feature selection. This AttributeEvaluator only evaluates one attribute.\n\n"
		+ "Current method: "+m_Measure+"\n\ncomputeCore: calculates those attributes that must appear in every valid reduct. This will only be used if the chosen" +
		"search method can use this information (i.e. HillClimber, AntSearch and PSOSearch).\n ------------------------------\n"
		+ "For more information see:\n\n"
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
		TechnicalInformation        result;
		TechnicalInformation        additional;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "R. Jensen, Q. Shen");
		result.setValue(Field.YEAR, "2009");
		result.setValue(Field.TITLE, "New Approaches to Fuzzy-rough Feature Selection. IEEE Transactions on Fuzzy Systems");
		result.setValue(Field.SCHOOL, "Aberystwyth University");
		result.setValue(Field.ADDRESS, "");

		additional = result.add(Type.INPROCEEDINGS);
		additional.setValue(Field.AUTHOR, "C. Cornelis, G. Hurtado Martin, R. Jensen and D. Slezak");
		additional.setValue(Field.TITLE, "Feature Selection with Fuzzy Decision Reducts");
		additional.setValue(Field.BOOKTITLE, "Third International Conference on Rough Sets and Knowledge Technology (RSKT'08)");
		additional.setValue(Field.YEAR, "2008");
		additional.setValue(Field.PAGES, "284-291");
		additional.setValue(Field.PUBLISHER, "Springer");

		additional = result.add(Type.INPROCEEDINGS);
		additional.setValue(Field.AUTHOR, "C. Cornelis and R. Jensen");
		additional.setValue(Field.TITLE, "A Noise-tolerant Approach to Fuzzy-Rough Feature Selection");
		additional.setValue(Field.BOOKTITLE, "17th International Conference on Fuzzy Systems (FUZZ-IEEE'08)");
		additional.setValue(Field.YEAR, "2008");
		additional.setValue(Field.PAGES, "1598-1605");
		additional.setValue(Field.PUBLISHER, "IEEE");

		return result;
	}

	/**
	 * Constructor
	 */
	public FuzzyRoughAttributeEval () {
		resetOptions();
	}


	/**
	 * Returns an enumeration describing the available options.
	 * @return an enumeration of all the available options.
	 *
	 **/
	public Enumeration<Option> listOptions () {
		Vector<Option> newVector = new Vector<Option>(4);
		newVector.addElement(new Option("\tSimilarity relation.", "R", 1, "-R <val>"));
		newVector.addElement(new Option("\tConnectives" + ".", "C", 1, "-C <val>"));
		newVector.addElement(new Option("\tComposition" 
				+ ".", "F", 1, "-F <val>"));
		newVector.addElement(new Option("\tMethod" 
				+ ".", "M", 1, "-M <val>"));
		newVector.addElement(new Option("\tStart with the core" 
				+ ".", "L", 0, "-L"));
		return  newVector.elements();
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

		resetOptions();
		String optionString;	   

		optionString = Utils.getOption('Z', options);
		if(optionString.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(optionString);
			if(nnSearchClassSpec.length == 0) { 
				throw new Exception("Invalid Measure specification string."); 
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setMeasure( (Measure) Utils.forName( Measure.class, className, nnSearchClassSpec) );
		}
		else {
			setMeasure(new WeakGamma());
		}


		optionString = Utils.getOption('I', options);
		if(optionString.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(optionString);
			if(nnSearchClassSpec.length == 0) { 
				throw new Exception("Invalid Implicator specification string."); 
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setImplicator( (Implicator) Utils.forName( Implicator.class, className, nnSearchClassSpec) );
		}
		else {
			setImplicator(new ImplicatorLukasiewicz());
		}


		optionString = Utils.getOption('T', options);
		if(optionString.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(optionString);
			if(nnSearchClassSpec.length == 0) { 
				throw new Exception("Invalid TNorm specification string."); 
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setTNorm( (TNorm) Utils.forName( TNorm.class, className, nnSearchClassSpec) );
		}
		else {
			setTNorm(new TNormLukasiewicz());
		}

		optionString = Utils.getOption('R', options);
		if(optionString.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(optionString);
			if(nnSearchClassSpec.length == 0) { 
				throw new Exception("Invalid Similarity specification string."); 
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setSimilarity( (Similarity) Utils.forName( Similarity.class, className, nnSearchClassSpec) );
		}
		else {
			setSimilarity(new Similarity3());
		}

		setUnsupervised(Utils.getFlag('U', options));
	}

	public void setMeasure(Measure fe) {
		m_Measure = fe;
	}

	public Measure getMeasure() {
		return m_Measure;
	}

	public void setUnsupervised(boolean u) {
		unsupervised=u;
	}

	public boolean getUnsupervised() {
		return unsupervised;
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



	/**
	 * Gets the current settings of FuzzyRoughSubsetEval
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions ()  {
		Vector<String>	result;

		result = new Vector<String>();

		result.add("-Z");
		result.add((m_Measure.getClass().getName() + " " +
				Utils.joinOptions(m_Measure.getOptions())).trim());

		result.add("-I");
		result.add((m_Implicator.getClass().getName() + " " +
				Utils.joinOptions(m_Implicator.getOptions())).trim());

		result.add("-T");
		result.add((m_TNorm.getClass().getName() + " " +
				Utils.joinOptions(m_TNorm.getOptions())).trim());

		result.add("-R");
		result.add((m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim());


		if (getUnsupervised()) {
			result.add("-U");
		}

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Returns the capabilities of this evaluator.
	 *
	 * @return            the capabilities of this evaluator
	 * @see               Capabilities
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

		return result;
	}

	/**
	 * Return a description of the fuzzy rough attribute evaluator.
	 *
	 * @return a description of the evaluator as a String.
	 */
	public String toString () {
		StringBuffer text = new StringBuffer();

		if (m_trainInstances == null) {
			text.append("FRFS feature evaluator has not been built yet\n");
		}
		else {
			text.append("\nFuzzy rough feature selection (one attribute only)\n\nMethod: "+m_Measure);
			if (m_Measure instanceof IVFuzzyMeasure) {

				text.append("\nParameter = "+((IVFuzzyMeasure)m_Measure).getParameter()+"\n");
			}
			text.append("\nSimilarity measure: "+m_Similarity);
			text.append("\nDecision similarity: "+m_DecisionSimilarity);
			text.append("\nImplicator: "+m_Implicator);
			text.append("\nT-Norm: "+m_TNorm+"\nRelation composition: "+m_Similarity.getTNorm());
			text.append("\n(S-Norm: "+m_SNorm+")\n\n");
			text.append("Dataset consistency: "+c_divisor+"\n");
		}

		return  text.toString();
	}

	public int[] postProcess (int[] attributeSet) throws Exception {
		return attributeSet;  
	}

	public boolean unsupervised=false;

	/**
	 * Generates an attribute evaluator. Has to initialise all fields of the 
	 * evaluator that are not being set via options.
	 *
	 *
	 * @param data set of instances serving as training data 
	 * @throws Exception if the evaluator has not been 
	 * generated successfully
	 */
	public void buildEvaluator (Instances data)	throws Exception {
		// can evaluator handle the data?
		getCapabilities().testWithFail(data);

		m_trainInstances = new Instances(data);
		//m_trainInstances.deleteWithMissingClass();
		
		m_numAttribs = m_trainInstances.numAttributes();
		m_numInstances = m_trainInstances.numInstances();

		//if the data has no decision feature, m_classIndex is negative
		m_classIndex = m_trainInstances.classIndex();
		
		//unsupervised
		if (m_classIndex<0 || unsupervised) {
			unsupervised=true;
			m_classIndex=-1;
		}
		else {
			m_isNumeric = m_trainInstances.attribute(m_classIndex).isNumeric();

			if (m_isNumeric) {
				m_DecisionSimilarity = m_Similarity;
			}
			else m_DecisionSimilarity = m_SimilarityEq;

		}
		
		m_Similarity.setInstances(m_trainInstances);
		m_DecisionSimilarity.setInstances(m_trainInstances);
		m_SimilarityEq.setInstances(m_trainInstances);
		m_composition = m_Similarity.getTNorm();

		m_Measure.set(m_Similarity,m_DecisionSimilarity,m_TNorm,m_composition,m_Implicator,m_SNorm,m_numInstances,m_numAttribs,m_classIndex,m_trainInstances);
		
		if (c_divisor==-1) c_divisor = getConsistency(); //determine the consistency of the full dataset
	}

	private final double getConsistency() throws Exception {
		BitSet full = new BitSet(m_numAttribs);
		for (int a = 0; a < m_numAttribs - 1; a++)
			if (a!=m_classIndex) full.set(a);


		double c_div = 0;

		if ((m_Measure instanceof FuzzyMutInf)||(m_Measure instanceof RoughMutInf)) {
			c_div=1;
		}
		else	c_div = m_Measure.calculate(full);


		if (c_div == 0 || c_div == Double.NaN) {
			System.err.println("\n*** Inconsistent data (full dataset value = "
					+ c_div + " for this measure) ***\n");
			c_div=1;
			//System.exit(1);
		}

		return c_div;
	}

	protected void resetOptions() {
		m_trainInstances = null;
		try {m_Similarity.setTNorm(new TNormLukasiewicz());}
		catch (Exception e) {}
	}



	/**
	 * Main method for testing this class.
	 *
	 * @param args the options
	 */
	public static void main (String[] args) {
		runEvaluator(new FuzzyRoughAttributeEval(), args);
	}

	/**
	 * Evaluates one attribute
	 */
	public double evaluateAttribute(int attribute) throws Exception {
		BitSet subset = new BitSet(m_numAttribs);
		subset.set(attribute);
		return m_Measure.calculate(subset);
	}
}

