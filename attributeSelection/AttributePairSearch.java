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
import weka.core.Utils;

import java.util.BitSet;
import java.util.Enumeration;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * AttributePairSearch :<br/>
 * <br/>
 * Compares each pair of attributes, and uses (fuzzy) dependency to gauge whether one attribute is subsumed by another.
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
public class AttributePairSearch 
extends ASSearch 
implements  OptionHandler {

	/** for serialization */
	static final long serialVersionUID = -6312951970168515471L;

	/** does the data have a class */
	protected boolean m_hasClass;

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	/** The number of attributes to select. -1 indicates that all attributes
      are to be retained. Has precedence over m_threshold */
	protected int m_numToSelect = -1;

	protected int m_calculatedNumToSelect;

	/** the merit of the best subset found */
	protected double m_bestMerit;

	/** the best subset found */
	protected BitSet m_best_group;
	protected ASEvaluation m_ASEval;

	protected Instances m_Instances;


	protected double alpha=1;

	/**
	 * Constructor
	 */
	public AttributePairSearch () {
		resetOptions();
	}

	/**
	 * Returns a string describing this search method
	 * @return a description of the search suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "AttributePairSearch:\n\nCompares each pair of attributes, and uses (fuzzy) dependency to gauge whether one attribute is subsumed by another.\n";
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

		

		optionString = Utils.getOption('N', options);
		if (optionString.length() != 0) {
			setNumToSelect(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setAlpha(Double.valueOf(optionString));
		}
	}

	/**
	 * Gets the current settings.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions () {
		String[] options = new String[12];
		int current = 0;


		options[current++] = "-N";
		options[current++] = ""+getNumToSelect();

		options[current++] = "-A";
		options[current++] = ""+getAlpha();

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
		FString.append("\tAttributePairSearch ");
		

		FString.append(searchOutput);
		
		return FString.toString();
	}

	private StringBuffer searchOutput;

	/**
	 * Searches the attribute subset space by forward selection.
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
		double temp_merit;
		BitSet temp_group;

		if (data != null) { // this is a fresh run so reset
			resetOptions();
			m_Instances = data;
		}		

		m_numAttribs = m_Instances.numAttributes();
		m_best_group = new BitSet(m_numAttribs);	
		m_ASEval = ASEval;
		
		USubsetEvaluator ASEvaluator = (USubsetEvaluator)m_ASEval;
		
		m_classIndex = m_Instances.classIndex();

		
		BitSet remove = new BitSet(m_numAttribs);
		remove.clear();
		temp_group = (BitSet)m_best_group.clone();	
		
		for (int a1=0;a1<m_numAttribs;a1++) {							
			if (a1 != m_classIndex && !remove.get(a1)) {
				temp_group.set(a1);
				
				for (i=0;i<m_numAttribs;i++) {
					if ((i != m_classIndex) && (!remove.get(i))&& i != a1) {
						temp_merit = ASEvaluator.evaluateSubset(temp_group,i);					
				
						if (temp_merit == 1) {
							remove.set(i);
						}
					}
				}
				
				temp_group.clear(a1);
			}

		}
		
		System.err.println("Removing "+remove.cardinality()+" attributes");
		
		for (int a=0;a<m_numAttribs;a++) {
			if (!remove.get(a)&&a!=m_classIndex) m_best_group.set(a);
		}
		
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
		m_best_group = null;
		m_ASEval = null;
		m_Instances = null;
		//alpha=1;
	}
}
