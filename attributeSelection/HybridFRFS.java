package weka.attributeSelection;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.BitSet;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * HybridWrapper :<br/>
 * <br/>
 * Performs a greedy forward or backward search through the space of attribute subsets. May start with no/all attributes or from an arbitrary point in the space. 
 * Stops when the addition/deletion of any remaining attributes results in a decrease in evaluation. Uses two measures to guide search, and a classifier to make the final decision at each level.<br/>
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -C
 *  Use conservative forward search</pre>
 * 
 * <pre> -B
 *  Use a backward search instead of a
 *  forward one.</pre>
 * 
 * <pre> -P &lt;start set&gt;
 *  Specify a starting set of attributes.
 *  Eg. 1,3,5-7.</pre>
 * 
 * <pre> -R
 *  Use the best encountered subset overall rather than the final subset.</pre>
 * 
 * <pre> -A &lt;threshold&gt;
 *  Specify a threshold (alpha) for alpha-decision reducts. Alpha should be in (0,1].
 * </pre>
 * 
 * <pre> -N &lt;num to select&gt;
 *  Specify number of attributes to select</pre>
 * 
 <!-- options-end -->
 *
 * @author Richard Jensen
 * @version $Revision: 1.9 $
 */
public class HybridFRFS  extends ASSearch 
implements OptionHandler {

	/** for serialization */
	static final long serialVersionUID = -6312951270168525471L;

	/** does the data have a class */
	protected boolean m_hasClass;

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	/** true if the user has requested a ranked list of attributes */
	protected boolean m_rankingRequested;

	/** 
	 * Use the best subset encountered, rather than the final subset
	 */
	protected boolean m_bestEncountered=false;

	/** used to indicate whether or not ranking has been performed */
	protected boolean m_doneRanking;


	/** The number of attributes to select. -1 indicates that all attributes
      are to be retained. Has precedence over m_threshold */
	protected int m_numToSelect = -1;

	protected int m_calculatedNumToSelect;

	/** the merit of the best subset found */
	protected double m_bestMerit;

	/** a ranked list of attribute indexes */
	protected double [][] m_rankedAtts;
	protected int m_rankedSoFar;

	/** the best subset found */
	protected BitSet m_best_group;

	public ASEvaluation m_evaluator1 = new FuzzyRoughSubsetEval();
	public ASEvaluation m_evaluator2 = new FuzzyRoughSubsetEval();

	/** holds the base classifier object */
	private Classifier m_BaseClassifier;

	/** holds an evaluation object */
	private Evaluation m_Evaluation;

	protected Instances m_Instances;

	/** Use a backwards search instead of a forwards one */
	protected boolean m_backward = false;

	/** If set then attributes will continue to be added during a forward
      search as long as the merit does not degrade */
	protected boolean m_conservativeSelection = false;

	protected double alpha=1;

	/**
	 * Constructor
	 */
	public HybridFRFS () {
		m_doneRanking = false;
		resetOptions();
	}

	/**
	 * Returns a string describing this search method
	 * @return a description of the search suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "HybridFRFS: Initially clusters features into groups based on correlation, and then performs feature selection using representatives from these groups.\n";
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String searchBackwardsTipText() {
		return "Search backwards rather than forwards.";
	}

	/**
	 * Set whether to search backwards instead of forwards
	 *
	 * @param back true to search backwards
	 */
	public void setSearchBackwards(boolean back) {
		m_backward = back;
		if (m_backward) {
			setBestEncountered(false);
		}
	}

	/**
	 * Get whether to search backwards
	 *
	 * @return true if the search will proceed backwards
	 */
	public boolean getSearchBackwards() {
		return m_backward;
	}


	/**
	 * Set the second subset evaluation measure
	 *
	 */
	public void setSubsetEvaluator2(ASEvaluation se) {
		m_evaluator2 = se;
	}

	/**
	 * Get the second subset evaluation measure
	 *
	 */
	public ASEvaluation getSubsetEvaluator2() {
		return m_evaluator2;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String thresholdTipText() {
		return "Set threshold by which attributes can be discarded. Default value "
				+ "results in no attributes being discarded. Use in conjunction with "
				+ "generateRanking";
	}


	public void setBestEncountered(boolean b) {
		m_bestEncountered = b;
	}


	public boolean getBestEncountered() {
		return m_bestEncountered;
	}


	public void setAlpha(double a) {
		alpha=a;
	}

	public double getAlpha() {
		return alpha;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String conservativeForwardSelectionTipText() {
		return "If true (and forward search is selected) then attributes "
				+"will continue to be added to the best subset as long as merit does "
				+"not degrade.";
	}

	/**
	 * Set whether attributes should continue to be added during
	 * a forward search as long as merit does not decrease
	 * @param c true if atts should continue to be atted
	 */
	public void setConservativeForwardSelection(boolean c) {
		m_conservativeSelection = c;
	}

	/**
	 * Gets whether conservative selection has been enabled
	 * @return true if conservative forward selection is enabled
	 */
	public boolean getConservativeForwardSelection() {
		return m_conservativeSelection;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String classifierTipText() {
		return "Classifier to use for estimating the accuracy of subsets";
	}

	/**
	 * Set the classifier to use for accuracy estimation
	 *
	 * @param newClassifier the Classifier to use.
	 */
	public void setClassifier (Classifier newClassifier) {
		m_BaseClassifier = newClassifier;
	}


	/**
	 * Get the classifier used as the base learner.
	 *
	 * @return the classifier used as the classifier
	 */
	public Classifier getClassifier () {
		return  m_BaseClassifier;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * @return an enumeration of all the available options.
	 **/
	public Enumeration<Option> listOptions () {
		Vector<Option> newVector = new Vector<Option>(6);

		newVector.addElement(new Option("\tUse conservative forward search"
				,"-C", 0, "-C"));

		newVector.addElement(new Option("\tUse a backward search instead of a"
				+"\n\tforward one."
				,"-B", 0, "-B"));
		newVector
		.addElement(new Option("\tSpecify a starting set of attributes." 
				+ "\n\tEg. 1,3,5-7."
				,"Z",1
				, "-Z <start set>"));

		newVector.addElement(new Option("\tProduce a ranked list of attributes."
				,"R",0,"-R"));
		newVector
		.addElement(new Option("\tSpecify a theshold by which attributes" 
				+ "\n\tmay be discarded from the ranking."
				+"\n\tUse in conjuction with -R","T",1
				, "-T <threshold>"));

		newVector
		.addElement(new Option("\tSpecify number of attributes to select" 
				,"N",1
				, "-N <num to select>"));

		newVector
		.addElement(new Option("\tAlpha" 
				,"A",1
				, "-A <value>"));

		return newVector.elements();

	}

	/**
	 * Parses a given list of options. <p/>
	 *
   <!-- options-start -->
	 * Valid options are: <p/>
	 * 
	 * <pre> -F
	 *  Second subset evaluator</pre>
	 *  
	 * <pre> -G
	 *  Classifier to decide which subset to select at each level of search.</pre>
	 *  
	 * <pre> -C
	 *  Use conservative forward search</pre>
	 * 
	 * <pre> -B
	 *  Use a backward search instead of a
	 *  forward one.</pre>
	 * 
	 * <pre> -P &lt;start set&gt;
	 *  Specify a starting set of attributes.
	 *  Eg. 1,3,5-7.</pre>
	 * 
	 * <pre> -R
	 *  Produce a ranked list of attributes.</pre>
	 * 
	 * <pre> -A &lt;threshold&gt;
	 *  Specify a threshold for subset 'goodness' - alpha
	 * </pre>
	 * 
	 * <pre> -N &lt;num to select&gt;
	 *  Specify number of attributes to select</pre>
	 * 
   <!-- options-end -->
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 */
	public void setOptions (String[] options)
			throws Exception {
		String optionString;
		resetOptions();

		setSearchBackwards(Utils.getFlag('B', options));

		setConservativeForwardSelection(Utils.getFlag('C', options));


		setBestEncountered(Utils.getFlag('R', options));

		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setAlpha(Double.valueOf(optionString));
		}


		optionString = Utils.getOption('F', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid SubsetEvaluator specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setSubsetEvaluator2( (ASEvaluation) Utils.forName( ASEvaluation.class, className, moreOptions) );
		}
		else {
			setSubsetEvaluator2(new FuzzyRoughSubsetEval());
		}
		optionString = Utils.getOption('G', options);
		if (optionString.length() == 0) optionString = IBk.class.getName();

		setClassifier(AbstractClassifier.forName(optionString, Utils.partitionOptions(options)));
	}

	/**
	 * Gets the current settings.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions () {
		String[] classifierOptions = new String[0];

		if ((m_BaseClassifier != null) && 
				(m_BaseClassifier instanceof OptionHandler)) {
			classifierOptions = ((OptionHandler)m_BaseClassifier).getOptions();
		}

		String[] options = new String[18+ classifierOptions.length];
		int current = 0;

		if (getSearchBackwards()) {
			options[current++] = "-B";
		}

		if (getConservativeForwardSelection()) {
			options[current++] = "-C";
		}



		if (getBestEncountered()) {
			options[current++] = "-R";
		}


		options[current++] = "-A";
		options[current++] = ""+getAlpha();

		options[current++] = "-F";
		options[current++] = ""+getSubsetEvaluator2().getClass().getName();

		if (getClassifier() != null) {
			options[current++] = "-G";
			options[current++] = getClassifier().getClass().getName();
		}

		options[current++] = "--";
		System.arraycopy(classifierOptions, 0, options, current, 
				classifierOptions.length);
		current += classifierOptions.length;

		while (current < options.length) {
			options[current++] = "";
		}

		return  options;
	}


	/**
	 * returns a description of the search.
	 * @return a description of the search as a String.
	 */
	public String toString() {
		StringBuffer FString = new StringBuffer();
		FString.append("\tHill Climber ("
				+ ((m_backward)
						? "backwards)"
								: "forwards)")+".\n\tStart set: ");

		if (m_backward) {
			FString.append("all attributes\n");
		} else {
			FString.append("no attributes\n");
		}
		if (!m_doneRanking) {
			FString.append("\tMerit of best subset found: "
					+Utils.doubleToString(Math.abs(m_bestMerit),8,3)+"\n");
		}

		FString.append(searchOutput);

		return FString.toString();
	}

	private StringBuffer searchOutput;

	private double[] evaluate(BitSet temp_group) throws Exception {
		double[] ret = new double[2];
		ret[0] = ((SubsetEvaluator)m_evaluator1).evaluateSubset(temp_group);
		ret[1] = ((SubsetEvaluator)m_evaluator2).evaluateSubset(temp_group);
		return ret;
	}


	/**
	 * Searches the attribute subset space via the two subset evaluators and classifier
	 *
	 * @param ASEval the attribute evaluator to guide the search
	 * @param data the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception if the search can't be completed
	 */
	public int[] search (ASEvaluation ASEval, Instances data) throws Exception {
		searchOutput = new StringBuffer("\n");

		if (m_bestEncountered) System.err.println("Using best encountered subset");

		int i;
		double best_merit = -500000000;
		double temp_best,temp_best2;
		double[] temp_merit = new double[2];
		int temp_index=0,temp_index2=0;
		BitSet temp_group;
		m_evaluator1=(ASEvaluation)ASEval;

		if (data != null) { // this is a fresh run so reset
			m_Instances = data;
		}

		m_numAttribs = m_Instances.numAttributes();

		if (m_best_group == null) {
			m_best_group = new BitSet(m_numAttribs);
		}

		m_hasClass = true;
		m_classIndex = m_Instances.classIndex();

		if (m_rankedAtts == null) {
			m_rankedAtts = new double[m_numAttribs][2];
			m_rankedSoFar = 0;
		}


		if (m_backward && m_rankedSoFar <= 0) {
			for (i = 0; i < m_numAttribs; i++) {
				if (i != m_classIndex) {
					m_best_group.set(i);
				}
			}

		}

		double[] prevEval = new double[2];
		prevEval[0]=0;
		prevEval[1]=0;
		m_evaluator1.buildEvaluator(data);
		m_evaluator2.buildEvaluator(data);

		// Evaluate the initial subset
		double[] ret = evaluate(m_best_group);
		//best_merit = eval(ret,prevEval);
		prevEval = ret;

		if (best_merit==1) {
			System.err.println(m_best_group+" => "+best_merit);
			return  attributeList(m_best_group);
		}

		// main search loop
		boolean done = false;
		boolean addone = false;
		boolean z,z2;
		String mes1="Measure 1";
		String mes2="Measure 2";

		if (m_evaluator1 instanceof FuzzyRoughSubsetEval) {
			mes1 = ((FuzzyRoughSubsetEval)m_evaluator1).getMeasure().toString();
			searchOutput.append("Measure 1: "+mes1+"\n");
			searchOutput.append("Consistency for measure 1: "+((FuzzyRoughSubsetEval)m_evaluator1).c_divisor+"\n");
		}

		if (m_evaluator2 instanceof FuzzyRoughSubsetEval) {
			mes2 = ((FuzzyRoughSubsetEval)m_evaluator2).getMeasure().toString();
			searchOutput.append("Measure 2: "+mes2+"\n");
			searchOutput.append("Consistency for measure 2: "+((FuzzyRoughSubsetEval)m_evaluator2).c_divisor+"\n");
		}

		BitSet bestEncounteredSubset=null;
		double bestErrorRate=-1000;

		while (!done) {
			temp_group = (BitSet)m_best_group.clone();

			if (m_backward) {
				temp_best = alpha;	
				temp_best2 = alpha;
			}
			else {
				//this enables search for non-monotonic measures
				temp_best=-1;
				temp_best2=-1;
				//temp_best = best_merit;
			}

			done = true;
			addone = false;

			for (i=0;i<m_numAttribs;i++) {
				if (m_backward) {
					z = ((i != m_classIndex) && (temp_group.get(i)));
				} else {
					z = ((i != m_classIndex) && (!temp_group.get(i)));
				}
				if (z) {
					// set/unset the bit
					if (m_backward) {
						temp_group.clear(i);
					} else {
						temp_group.set(i);
					}

					//evaluate the subset via the two measures
					double[] tempEval = evaluate(temp_group);

					//System.err.println(temp_group);

					if (m_backward) {
						z = (tempEval[0] >= temp_best);
						z2 = (tempEval[1] >= temp_best2);
					} else {
						if (m_conservativeSelection) {
							z = (tempEval[0] >= temp_best);
							z2 = (tempEval[1] >= temp_best2);
						} else {
							z = (tempEval[0] > temp_best);
							z2 = (tempEval[1] > temp_best2);
						}
					}

					if (z) {
						temp_best = tempEval[0];
						temp_index = i;
						addone = true;
						done = false;
					}
					if (z2) {
						temp_best2 = tempEval[1];
						temp_index2 = i;
						addone = true;
						done = false;
					}

					// unset this addition/deletion
					if (m_backward) {
						temp_group.set(i);
					} else {
						temp_group.clear(i);
					}

				}
			}
			if (addone) {

				/**
				 * 
				 * Use the classifier to choose the best subset
				 * between measure 1 and measure 2...
				 * 
				 */
				double eval1=0;
				double eval2=0;

				if (m_backward) {
					//if (temp_index!=temp_index2) {
					m_best_group.clear(temp_index);
					eval1 = eval(data,m_best_group);
					m_best_group.set(temp_index);

					m_best_group.clear(temp_index2);
					eval2 = eval(data,m_best_group);
					m_best_group.set(temp_index2);
					//}

					if (eval1>eval2) {m_best_group.clear(temp_index);best_merit=eval1;}
					else {m_best_group.clear(temp_index2);best_merit=eval2;}

				} else {
					m_best_group.set(temp_index);
					eval1 = eval(data,m_best_group);
					System.err.println(mes1+": "+m_best_group+" => "+eval1);
					m_best_group.clear(temp_index);


					m_best_group.set(temp_index2);
					eval2 = eval(data,m_best_group);
					System.err.println(mes2+": "+m_best_group+" => "+eval2);
					m_best_group.clear(temp_index2);

					if (eval1>eval2) {m_best_group.set(temp_index);best_merit=eval1;}
					else {m_best_group.set(temp_index2);best_merit=eval2;}

					System.err.println("Choosing "+m_best_group+" => "+best_merit+"\n");

					if (temp_best>=alpha) {
						done=true;
						break;
					}
				}

				if (best_merit>bestErrorRate) {
					bestEncounteredSubset = (BitSet)m_best_group.clone();
					bestErrorRate = best_merit;
				}

			}
		}
		m_bestMerit = best_merit;

		//return bestEncountered to return best encountered subset for the classifier rather than the final subset
		if (m_bestEncountered) {
			System.err.println("Best encountered: "+bestEncounteredSubset);
			m_best_group = bestEncounteredSubset;
			m_bestMerit = bestErrorRate;

		}

		return attributeList(m_best_group);
	}

	private final double eval(Instances m_trainInstances, BitSet subset) throws Exception {
		double errorRate = 0;
		double[] repError = new double[5];
		int numAttributes = 0;
		int i, j;

		Random Rnd = new Random();
		Remove delTransform = new Remove();
		delTransform.setInvertSelection(true);
		// copy the instances
		Instances trainCopy = new Instances(m_trainInstances);

		// count attributes set in the BitSet
		for (i = 0; i < m_numAttribs; i++) {
			if (subset.get(i)) {
				numAttributes++;
			}
		}

		// set up an array of attribute indexes for the filter (+1 for the class)
		int[] featArray = new int[numAttributes + 1];

		for (i = 0, j = 0; i < m_numAttribs; i++) {
			if (subset.get(i)) {
				featArray[j++] = i;
			}
		}

		featArray[j] = m_classIndex;
		delTransform.setAttributeIndicesArray(featArray);
		delTransform.setInputFormat(trainCopy);
		trainCopy = Filter.useFilter(trainCopy, delTransform);

		// 5 repetitions of 10-fold cross validation
		for (i = 0; i < 5; i++) {
			m_Evaluation = new Evaluation(trainCopy);
			m_Evaluation.crossValidateModel(m_BaseClassifier, trainCopy, 10, Rnd);
			repError[i] = m_Evaluation.errorRate();
			errorRate += repError[i];
		}

		errorRate /= 5.0;
		m_Evaluation = null;
		return -errorRate;
	}

	/**
	 * converts a BitSet into a list of attribute indexes 
	 * @param group the BitSet to convert
	 * @return an array of attribute indexes
	 **/
	protected int[] attributeList (BitSet group) {
		int count = 0;

		// count how many were selected
		for (int i = 0; i < m_numAttribs; i++) {
			if (group.get(i)) {
				count++;
			}
		}

		int[] list = new int[count];
		count = 0;

		for (int i = 0; i < m_numAttribs; i++) {
			if (group.get(i)) {
				list[count++] = i;
			}
		}

		return  list;
	}

	/**
	 * Resets options
	 */
	protected void resetOptions() {
		m_bestEncountered = false;
		m_best_group = null;
		m_Instances = null;
		m_rankedSoFar = -1;
		m_rankedAtts = null;
		m_BaseClassifier=new IBk();
		//alpha=1;
	}
}
