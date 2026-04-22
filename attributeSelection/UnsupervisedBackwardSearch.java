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
 *    HillClimber.java
 *   
 *
 */


import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.fuzzy.unsupervised.UnsupervisedFuzzyMeasure;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * UnsupervisedBackwardSearch :<br/>
 * <br/>
 * Performs a linear backward search through the space of attribute subsets. Starts with all attributes. 
 * Stops when the deletion of any remaining attributes results in a decrease in evaluation. <br/>
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * 
 * <pre> -A &lt;threshold&gt;
 *  Specify a threshold (alpha) for alpha-decision reducts. Alpha should be in (0,1].
 * </pre>
 * 
 * 
 <!-- options-end -->
 *
 * @author Richard Jensen
 * @version $Revision: 1.9 $
 */
public class UnsupervisedBackwardSearch 
extends ASSearch 
implements  OptionHandler {

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

	/**
	 * A threshold by which to discard attributes---used by the
	 * AttributeSelection module
	 */
	protected double m_threshold;

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
	protected ASEvaluation m_ASEval;

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
	public boolean m_debug=false;

	//flag to determine whether the order of the ranking should be reversed
	public boolean m_reverseOrder = false;


	/**
	 * Constructor
	 */
	public UnsupervisedBackwardSearch () {
		m_threshold = -Double.MAX_VALUE;
		m_doneRanking = false;
		m_reverseOrder = false;
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
		return "Backward linear search :\n\nPerforms a linear backward search "
		+"through "
		+"the space of attribute subsets. May start with no/all attributes or from "
		+"an arbitrary point in the space. Stops when the addition/deletion of any "
		+"remaining attributes results in a decrease in evaluation. \n";
	}


	public void setAlpha(double a) {
		alpha=a;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setDebug(boolean flag) {
		m_debug=flag;
	}

	public boolean getDebug() {
		return m_debug;
	}

	public void setReverseOrder(boolean b) {
		m_reverseOrder=b;
	}

	public boolean getReverseOrder() {
		return m_reverseOrder;
	}

	/**
	 * Returns an enumeration describing the available options.
	 * @return an enumeration of all the available options.
	 **/
	public Enumeration<Option> listOptions () {
		Vector<Option> newVector = new Vector<Option>(6);


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

		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setAlpha(Double.valueOf(optionString));
		}

		optionString = Utils.getOption('F', options);
		if (optionString.length() != 0)
			setRanking(new SelectedTag(Integer.parseInt(optionString), TAGS_METHOD));
		else
			setRanking(new SelectedTag(0, TAGS_METHOD));


		setDebug(Utils.getFlag('L', options));
		setReverseOrder(Utils.getFlag('S', options));
	}

	/**
	 * Gets the ranking method
	 *
	 * @return the function.
	 */
	public SelectedTag getRanking() {
		return new SelectedTag(m_Ranking, TAGS_METHOD);
	}

	/**
	 * Sets the ranking method
	 *
	 * @param value the ranking method
	 */
	public void setRanking(SelectedTag value) {
		if (value.getTags() == TAGS_METHOD)
			m_Ranking = value.getSelectedTag().getID();
	}

	/**
	 * Gets the current settings.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions () {
		String[] options = new String[11];
		int current = 0;

		options[current++] = "-A";
		options[current++] = ""+getAlpha();
		
		options[current++] = "-F";
		options[current++] = ""+m_Ranking;

		if (getDebug()) {
			options[current++]="-L";
		}

		if (getReverseOrder()) {
			options[current++]="-S";
		}

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
		FString.append("Unsupervised backward elimination.\n\tStart set: all attributes");




		if (!m_doneRanking) {
			FString.append("\tMerit of best subset found: "
					+Utils.doubleToString(Math.abs(m_bestMerit),8,3)+"\n");
		}

		if ((m_threshold != -Double.MAX_VALUE) && (m_doneRanking)) {
			FString.append("\tThreshold for discarding attributes: "
					+ Utils.doubleToString(m_threshold,8,4)+"\n");
		}

		FString.append(searchOutput);

		return FString.toString();
	}

	private StringBuffer searchOutput;

	public class Index implements Comparable<Index> {
		int value; double entropy=0;

		public Index(int v, double d) {
			value=v;
			entropy=d;
		}

		//for sorting in ascending worth of attributes (i.e. decreasing entropy)
		public int compareTo(Index i) {
			return entropy > i.entropy ? -1 : entropy < i.entropy ? 1 : 0;
		}

	}

	public int m_Ranking=0;

	public static final int NO_RANKING=0;
	public static final int FUZZY_ENTROPY=1;
	public static final int GRANULARITY=2;

	public static final Tag [] TAGS_METHOD = {
		new Tag(NO_RANKING, "No ranking"),
		new Tag(FUZZY_ENTROPY, "Fuzzy entropy"),
		new Tag(GRANULARITY, "Granularity"),
	};

	/**
	 * Searches the attribute subset space by forward selection.
	 *
	 * @param ASEval the attribute evaluator to guide the search
	 * @param data the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception if the search can't be completed
	 */
	public int[] search (ASEvaluation ASEval, Instances data) throws Exception {
		searchOutput = new StringBuffer("\n");
		double best_merit = -Double.MAX_VALUE;
		double temp_merit;

		BitSet temp_group;

		if (data != null) { // this is a fresh run so reset
			resetOptions();
			m_Instances = data;
		}
		m_ASEval = ASEval;

		m_numAttribs = m_Instances.numAttributes();

		if (m_best_group == null) {
			m_best_group = new BitSet(m_numAttribs);
		}

		if (!(m_ASEval instanceof USubsetEvaluator)) {
			throw  new Exception(m_ASEval.getClass().getName() 
					+ " is not an unsupervised " 
					+ "Subset evaluator!");
		}

		m_startRange.setUpper(m_numAttribs-1);		
		m_classIndex = m_Instances.classIndex();


		USubsetEvaluator ASEvaluator = (USubsetEvaluator)m_ASEval;


		for (int i = 0; i < m_numAttribs; i++) {
			if (i != m_classIndex) {
				m_best_group.set(i);
			}
		}

		ArrayList<Index> order = new ArrayList<Index>(m_numAttribs);

		if (m_Ranking!=0) {
			UnsupervisedFuzzyMeasure measure = ASEvaluator.getFuzzyMeasure();

			for (int a=0;a<m_numAttribs;a++) {
				if (a!=m_classIndex) {
					double eval=-1;

					switch (m_Ranking) {
					case FUZZY_ENTROPY:
						eval = measure.calculateEntropy(a);
						break;
					case GRANULARITY:
						eval = measure.calculateGranularity(a);
						break;
					}

					order.add(new Index(a,eval));
				}
			}
		}
		else { //no ranking
			for (int a=0;a<m_numAttribs;a++) {
				if (a!=m_classIndex) {	
					order.add(new Index(a,a));
				}
			}
		}

		Collections.sort(order);

		if (m_reverseOrder) {
			Collections.reverse(order);
		}

		Iterator<Index> it = order.iterator();

		temp_group = (BitSet)m_best_group.clone();				

		while (it.hasNext()) {
			int i = (it.next()).value;

			temp_group.clear(i);
			temp_merit = ASEvaluator.evaluateSubset(temp_group,i);
			if (m_debug) System.err.println(temp_group+": "+temp_merit);

			if (temp_merit >= alpha) {
				best_merit = temp_merit;

				m_best_group.clear(i);
				searchOutput.append(m_best_group+" => "+best_merit+"\n");
				if (m_debug) System.err.println(m_best_group+" => "+best_merit);
			}
			else temp_group.set(i);
		}

		/*
		temp_group = (BitSet)m_best_group.clone();				

		for (int i=m_numAttribs-1;i>=0;i--) {
			if ((i != m_classIndex) && (temp_group.get(i))) {

				temp_group.clear(i);
				temp_merit = ASEvaluator.evaluateSubset(temp_group,i);

				if (temp_merit >= alpha) {
					best_merit = temp_merit;

					m_best_group.clear(i);
					searchOutput.append(m_best_group+" => "+best_merit+"\n");
					if (m_debug) System.err.println(m_best_group+" => "+best_merit);
				}
				else temp_group.set(i);


			}
		}*/

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
		m_ASEval = null;
		m_Instances = null;
		m_rankedSoFar = -1;
		m_rankedAtts = null;
		//alpha=1;
	}
}

