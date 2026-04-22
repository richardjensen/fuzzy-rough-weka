package weka.attributeSelection;

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
 *    CombinedMeasureSearch.java
 *   
 *
 */


import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.Utils;

import java.util.BitSet;
import java.util.Enumeration;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * CombinedMeasureSearch :<br/>
 * <br/>
 * Performs a greedy forward or backward search through the space of attribute subsets. May start with no/all attributes or from an arbitrary point in the space. 
 * Stops when the addition/deletion of any remaining attributes results in a decrease in evaluation. Uses two measures to guide search.<br/>
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
 *  Produce a ranked list of attributes.</pre>
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
public class CombinedMeasureSearch 
extends ASSearch 
implements StartSetHandler, OptionHandler {

	/** for serialization */
	static final long serialVersionUID = -6312951970168525471L;

	/** does the data have a class */
	protected boolean m_hasClass;

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	/** true if the user has requested a ranked list of attributes */
	protected boolean m_rankingRequested;

	/** 
	 * go from one side of the search space to the other in order to generate
	 * a ranking
	 */
	protected boolean m_doRank;

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

	protected Instances m_Instances;

	/** holds the start set for the search as a Range */
	protected Range m_startRange;

	/** holds an array of starting attributes */
	protected int [] m_starting;

	/** Use a backwards search instead of a forwards one */
	protected boolean m_backward = false;

	/** If set then attributes will continue to be added during a forward
      search as long as the merit does not degrade */
	protected boolean m_conservativeSelection = false;

	protected double alpha=1;

	/**
	 * Constructor
	 */
	public CombinedMeasureSearch () {
		m_doneRanking = false;
		m_startRange = new Range();
		m_starting = null;
		resetOptions();
	}

	/**
	 * Returns a string describing this search method
	 * @return a description of the search suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "CombinedMeasureSearch :\n\nPerforms a greedy forward or backward search "
		+"through "
		+"the space of attribute subsets. May start with no/all attributes or from "
		+"an arbitrary point in the space. Uses two measures to guide search. Stops when the addition/deletion of any "
		+"remaining attributes results in a decrease in evaluation.\n";
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
			setGenerateRanking(false);
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



	public void setSubsetEvaluator2(ASEvaluation se) {
		m_evaluator2 = se;
	}


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


	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String numToSelectTipText() {
		return "Specify the number of attributes to retain. The default value "
		+"(-1) indicates that all attributes are to be retained. Use either "
		+"this option or a threshold to reduce the attribute set.";
	}

	/**
	 * Specify the number of attributes to select from the ranked list
	 * (if generating a ranking). -1
	 * indicates that all attributes are to be retained.
	 * @param n the number of attributes to retain
	 */
	public void setNumToSelect(int n) {
		m_numToSelect = n;
	}

	/**
	 * Gets the number of attributes to be retained.
	 * @return the number of attributes to retain
	 */
	public int getNumToSelect() {
		return m_numToSelect;
	}

	/**
	 * Gets the calculated number of attributes to retain. This is the
	 * actual number of attributes to retain. This is the same as
	 * getNumToSelect if the user specifies a number which is not less
	 * than zero. Otherwise it should be the number of attributes in the
	 * (potentially transformed) data.
	 */
	public int getCalculatedNumToSelect() {
		if (m_numToSelect >= 0) {
			m_calculatedNumToSelect = m_numToSelect;
		}
		return m_calculatedNumToSelect;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String generateRankingTipText() {
		return "Set to true if a ranked list is required.";
	}

	/**
	 * Records whether the user has requested a ranked list of attributes.
	 * @param doRank true if ranking is requested
	 */
	public void setGenerateRanking(boolean doRank) {
		m_rankingRequested = doRank;
	}

	/**
	 * Gets whether ranking has been requested. This is used by the
	 * AttributeSelection module to determine if rankedAttributes()
	 * should be called.
	 * @return true if ranking has been requested.
	 */
	public boolean getGenerateRanking() {
		return m_rankingRequested;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String startSetTipText() {
		return "Set the start point for the search. This is specified as a comma "
		+"seperated list off attribute indexes starting at 1. It can include "
		+"ranges. Eg. 1,2,5-9,17.";
	}

	/**
	 * Sets a starting set of attributes for the search. It is the
	 * search method's responsibility to report this start set (if any)
	 * in its toString() method.
	 * @param startSet a string containing a list of attributes (and or ranges),
	 * eg. 1,2,6,10-15.
	 * @throws Exception if start set can't be set.
	 */
	public void setStartSet (String startSet) throws Exception {
		m_startRange.setRanges(startSet);
	}

	/**
	 * Returns a list of attributes (and or attribute ranges) as a String
	 * @return a list of attributes (and or attribute ranges)
	 */
	public String getStartSet () {
		return m_startRange.getRanges();
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

		optionString = Utils.getOption('Z', options);
		if (optionString.length() != 0) {
			setStartSet(optionString);
		}

		setGenerateRanking(Utils.getFlag('R', options));


		optionString = Utils.getOption('N', options);
		if (optionString.length() != 0) {
			setNumToSelect(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setAlpha(Double.valueOf(optionString));
		}


		optionString = Utils.getOption('F', options);
		if(optionString.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(optionString);
			if(nnSearchClassSpec.length == 0) { 
				throw new Exception("Invalid SubsetEvaluator specification string."); 
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setSubsetEvaluator2( (ASEvaluation) Utils.forName( ASEvaluation.class, className, nnSearchClassSpec) );
		}
		else {
			setSubsetEvaluator2(new FuzzyRoughSubsetEval());
		}
	}

	/**
	 * Gets the current settings.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions () {
		String[] options = new String[15];
		int current = 0;

		if (getSearchBackwards()) {
			options[current++] = "-B";
		}

		if (getConservativeForwardSelection()) {
			options[current++] = "-C";
		}

		if (!(getStartSet().equals(""))) {
			options[current++] = "-Z";
			options[current++] = ""+startSetToString();
		}

		if (getGenerateRanking()) {
			options[current++] = "-R";
		}

		options[current++] = "-N";
		options[current++] = ""+getNumToSelect();

		options[current++] = "-A";
		options[current++] = ""+getAlpha();

		options[current++] = "-F";
		options[current++] = ""+getSubsetEvaluator2().getClass().getName();


		while (current < options.length) {
			options[current++] = "";
		}
		return  options;
	}

	/**
	 * converts the array of starting attributes to a string. This is
	 * used by getOptions to return the actual attributes specified
	 * as the starting set. This is better than using m_startRanges.getRanges()
	 * as the same start set can be specified in different ways from the
	 * command line---eg 1,2,3 == 1-3. This is to ensure that stuff that
	 * is stored in a database is comparable.
	 * @return a comma separated list of individual attribute numbers as a String
	 */
	protected String startSetToString() {
		StringBuffer FString = new StringBuffer();
		boolean didPrint;

		if (m_starting == null) {
			return getStartSet();
		}
		for (int i = 0; i < m_starting.length; i++) {
			didPrint = false;

			if ((m_hasClass == false) || 
					(m_hasClass == true && i != m_classIndex)) {
				FString.append((m_starting[i] + 1));
				didPrint = true;
			}

			if (i == (m_starting.length - 1)) {
				FString.append("");
			}
			else {
				if (didPrint) {
					FString.append(",");
				}
			}
		}

		return FString.toString();
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

		if (m_starting == null) {
			if (m_backward) {
				FString.append("all attributes\n");
			} else {
				FString.append("no attributes\n");
			}
		}
		else {
			FString.append(startSetToString()+"\n");
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

	//How to combine the two measures
	private double eval(double[] temp_merit) {
		return temp_merit[0]*temp_merit[1]; 
	}

	/**
	 * Searches the attribute subset space (hillclimbing).
	 *
	 * @param ASEval the attribute evaluator to guide the search
	 * @param data the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception if the search can't be completed
	 */
	public int[] search (ASEvaluation ASEval, Instances data)
	throws Exception {
		searchOutput = new StringBuffer("\n");
		int i;
		double best_merit = -500000000;
		double temp_best;
		double[] temp_merit = new double[2];
		int temp_index=0;
		BitSet temp_group;
		m_evaluator1=(ASEvaluation)ASEval;

		if (data != null) { // this is a fresh run so reset
			resetOptions();
			m_Instances = data;
		}


		m_numAttribs = m_Instances.numAttributes();

		if (m_best_group == null) {
			m_best_group = new BitSet(m_numAttribs);
		}

		m_startRange.setUpper(m_numAttribs-1);
		if (!(getStartSet().equals(""))) {
			m_starting = m_startRange.getSelection();
		}

		m_hasClass = true;
		m_classIndex = m_Instances.classIndex();

		if (m_rankedAtts == null) {
			m_rankedAtts = new double[m_numAttribs][2];
			m_rankedSoFar = 0;
		}

		// If a starting subset has been supplied, then initialise the bitset
		if (m_starting != null && m_rankedSoFar <= 0) {
			for (i = 0; i < m_starting.length; i++) {
				if ((m_starting[i]) != m_classIndex) {
					m_best_group.set(m_starting[i]);
				}
			}
		} else {
			if (m_backward && m_rankedSoFar <= 0) {
				for (i = 0; i < m_numAttribs; i++) {
					if (i != m_classIndex) {
						m_best_group.set(i);
					}
				}
			}
		}

		
		m_evaluator1.buildEvaluator(data);
		m_evaluator2.buildEvaluator(data);

		// Evaluate the initial subset
		double[] ret = evaluate(m_best_group);
		best_merit = eval(ret);

		if (best_merit==1) {
			System.err.println(m_best_group+" => "+best_merit);
			return  attributeList(m_best_group);
		}

		// main search loop
		boolean done = false;
		boolean addone = false;
		boolean z;

		String mes1="Measure 1";
		String mes2="Measure 2";
		
		if (m_evaluator1 instanceof FuzzyRoughSubsetEval) {
			mes1 = ((FuzzyRoughSubsetEval)m_evaluator1).getMeasure().toString();
			searchOutput.append("Measure 1: "+mes1+"\n");
			System.err.println("Measure 1: "+mes1);
			searchOutput.append("Consistency for measure 1: "+((FuzzyRoughSubsetEval)m_evaluator1).c_divisor+"\n");
			System.err.println("Consistency for measure 1: "+((FuzzyRoughSubsetEval)m_evaluator1).c_divisor);
		}
		if (m_evaluator2 instanceof FuzzyRoughSubsetEval) {
			mes2 = ((FuzzyRoughSubsetEval)m_evaluator2).getMeasure().toString();
			searchOutput.append("Measure 2: "+mes2+"\n");
			System.err.println("Measure 2: "+mes2);
			searchOutput.append("Consistency for measure 2: "+((FuzzyRoughSubsetEval)m_evaluator2).c_divisor+"\n\n");
			System.err.println("Consistency for measure 2: "+((FuzzyRoughSubsetEval)m_evaluator2).c_divisor);
		}

		searchOutput.append("Attr.\t"+mes2+"\t"+mes1+"\n");
		while (!done) {
			temp_group = (BitSet)m_best_group.clone();

			if (m_backward) {
				temp_best = alpha;				
			}
			else {
				//this enables search for non-monotonic measures
				temp_best=-1;
				//temp_best = best_merit;
			}

			if (m_doRank) {
				temp_best = -Double.MAX_VALUE;
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

					double[] tempEval = evaluate(temp_group);
					double eval = eval(tempEval);

					//System.err.println(temp_group);

					if (m_backward) {
						z = (eval >= temp_best);
					} else {
						if (m_conservativeSelection) {
							z = (eval >= temp_best);
						} else {
							z = (eval > temp_best);
						}
					}

					if (z) {
						temp_best = eval;
						temp_merit = tempEval;
						temp_index = i;
						addone = true;
						done = false;
					}

					// unset this addition/deletion
					if (m_backward) {
						temp_group.set(i);
					} else {
						temp_group.clear(i);
					}
					if (m_doRank) {
						done = false;
					}
				}
			}
			if (addone) {
				best_merit = temp_best;

				if (m_backward) {
					m_best_group.clear(temp_index);
					searchOutput.append(m_best_group+" => "+best_merit+"\n");
					System.err.println(m_best_group+" => "+best_merit);
				} else {
					m_best_group.set(temp_index);
					double gamma = ((SubsetEvaluator)m_evaluator1).evaluateSubset(m_best_group);

					//searchOutput.append(m_best_group+" => "+best_merit+"\n");
					searchOutput.append(temp_index+"\t"+best_merit+"\t"+gamma+"\n");
					System.err.println(temp_index);
					System.err.println("with gamma = "+gamma+"\n\n");

					//int val = (int)(100*(double)best_merit);
					//setProgressBarValue(val);
					
					if (best_merit>=alpha) {
						done=true;
						break;
					}
				}

				m_rankedAtts[m_rankedSoFar][0] = temp_index;
				m_rankedAtts[m_rankedSoFar][1] = best_merit;
				m_rankedSoFar++;
			}
		}
		m_bestMerit = best_merit;

		return attributeList(m_best_group);
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
		m_doRank = false;
		m_best_group = null;
		m_Instances = null;
		m_rankedSoFar = -1;
		m_rankedAtts = null;
		//alpha=1;
	}
}
