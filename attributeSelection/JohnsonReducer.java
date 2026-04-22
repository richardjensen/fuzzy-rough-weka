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
import java.util.ArrayList;

import weka.fuzzy.measure.*;
import weka.fuzzy.snorm.*;
import weka.fuzzy.implicator.*;

/**  SATSearch
 *
 * @author Richard Jensen
 * @version $Revision: 1.9 $
 */
public class JohnsonReducer 
extends ASSearch 
implements  OptionHandler {

	/** for serialization */
	static final long serialVersionUID = -6312951990168525471L;
	long start;

	/** does the data have a class */
	protected boolean m_hasClass;

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	/** the merit of the best subset found */
	protected double m_bestMerit;
	protected Implicator m_Implicator;

	/** the best subset found */
	protected BitSet m_best_group;
	protected ASEvaluation m_ASEval;

	protected Instances m_Instances;

	SNorm m_SNorm;
	boolean skip=false;
	ArrayList<Clause> newCL;
	ArrayList<Clause> cands,cands2=null;
	int bound = 100; //depth bound

	protected double alpha=1;

	/**
	 * Constructor
	 */
	public JohnsonReducer () {
		resetOptions();
	}

	/**
	 * Returns a string describing this search method
	 * @return a description of the search suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Standard Johnson Reducer - hill-climber that iteratively chooses the most commonly-occuring feature in the clauses. \n\nNote: the FDM measure must be used in conjunction with FuzzyRoughSubset eval for this search.\n";
	}


	public void setAlpha(double a) {
		alpha=a;
	}

	public double getAlpha() {
		return alpha;
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
	public void setOptions (String[] options) throws Exception {
		String optionString;
		resetOptions();

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
		FString.append("\tSAT-based search."
				+" ");

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
	public int[] search (ASEvaluation ASEval, Instances data) throws Exception {
		searchOutput = new StringBuffer("\n");

		if (data != null) { // this is a fresh run so reset
			resetOptions();
			m_Instances = data;
		}
		m_ASEval = ASEval;

		m_numAttribs = m_Instances.numAttributes();

		if (m_best_group == null) {
			m_best_group = new BitSet(m_numAttribs);
		}

		if (!(m_ASEval instanceof FuzzyRoughSubsetEval)) {

			if (m_ASEval instanceof FilteredSubsetEval) {			
				m_ASEval = ((FilteredSubsetEval) m_ASEval).getSubsetEvaluator();

				if (!(m_ASEval instanceof FuzzyRoughSubsetEval)) {
					throw  new Exception(m_ASEval.getClass().getName() + " is not a FuzzyRoughSubsetEval evaluator!");

				}
			}
			else 
				throw  new Exception(m_ASEval.getClass().getName() 		+ " is not a FuzzyRoughSubsetEval evaluator!");
		}


		reduct = new BitSet(m_numAttribs);

		m_hasClass = true;
		m_classIndex = m_Instances.classIndex();


		FuzzyRoughSubsetEval ev = (FuzzyRoughSubsetEval)m_ASEval;
		FDM fdm=null;

		if (ev.getMeasure() instanceof FDM) {
			fdm = (FDM)ev.getMeasure();
		}
		else {
			throw new Exception("You must use the FDM measure with this search method");
		}

		cands = fdm.getClauses();
		searchOutput.append("\nNumber of clauses used: "+cands.size()+"\n-----------------------------------\nDiscovered reducts: \n");
		m_SNorm = fdm.m_composition.getAssociatedSNorm();
		m_Implicator = fdm.m_Implicator;


		//sort the clauses in order of size
		java.util.Collections.sort(cands);

		start = System.currentTimeMillis();

		reduct = new BitSet(m_numAttribs);
		//reduct = (BitSet) fdm.core.clone();

		while (cands.size()>0) {

			int var = heuristicPick(cands);
			reduct.set(var);
			cands = updateTrue(cands);
		}


		float time = (float)(System.currentTimeMillis() - start)/1000;
		System.err.println("Finished search: "+time+" s");
		searchOutput.append("Finished search: "+time+" s");
		//print(cands);

		return attributeList(reduct);
	}

	BitSet toRemove=null;



	BitSet reduct=null;
	//keep track of those feature that have been selected and unselected
	ArrayList<Clause> newClauses;



	//The heuristic method. Evaluates the features appearing in the
	//given the list of clauses, ArrayList CL, and returns the integer
	//corresponding to the best feature chosen. 
	public final int heuristicPick(ArrayList<Clause> CL) {
		int ret=-1; //the chosen attr
		double[] occur = new double[m_numAttribs]; //occurrences of the features

		java.util.Iterator<Clause> i = CL.iterator();
		boolean unit=false;
		double currBest=-100000000;
		Clause t;

		//for each clause
		while (i.hasNext()) {
			t =  i.next();

			//if the clause is of size 1, then the feature appearing
			//in it must be set to true
			if (t.setValues==1) {					
				int attr=-1;

				//find the unit variable
				for (int a = reduct.nextClearBit(0); a < m_numAttribs; a = reduct.nextClearBit(a + 1)) {	
					if (a!=m_classIndex && t.getVariableValue(a)>0) {
						attr=a;
						break;
					}
				}

				unit=true;
				return attr;
			}
			else { //the clause contains more than one feature

				for (int a = reduct.nextClearBit(0); a < m_numAttribs; a = reduct.nextClearBit(a + 1)) {
					if (a!=m_classIndex) {
						occur[a]+=t.getVariableValue(a);//1/t.cardinality;
					}
				}
			}
		}


		//if not a unit clauses...
		if (!unit) {
			currBest=-100000000;

			//find the best feature (represented by an integer)
			for (int a = reduct.nextClearBit(0); a < m_numAttribs; a = reduct.nextClearBit(a + 1)) {	
				if (a!=m_classIndex&&occur[a]>currBest) {
					currBest = occur[a];
					ret=a;
				}

			}
		}

		//return the best feature
		return ret;
	}

	//Update the clause database given that variable 'var' has just
	//been set to true (i.e. it has just been selected)
	// - all clauses containing 'var' will be removed
	public final ArrayList<Clause> updateTrue(ArrayList<Clause> CL) {
		newCL = new ArrayList<Clause>();
		java.util.Iterator<Clause> i = CL.iterator();
		Clause t;

		while(i.hasNext()) {
			t =  i.next();
			if (!isSatisfied(t)) newCL.add(t.clone()); //this has to be t.clone()	
		}

		return newCL;
	}


	/**
	 * Get the current degree of satisfaction of the Clause t based on the selected attributes
	 * 
	 */
	private boolean isSatisfied(Clause t) {
		double max=0;

		for (int a = reduct.nextSetBit(0); a >= 0; a = reduct.nextSetBit(a + 1)) {	
			if (a!=m_classIndex) {
				max = m_SNorm.calculate(max, t.getVariableValue(a));
				if (max>=1) break;
			}
		}

		max = m_Implicator.calculate(t.getVariableValue(m_classIndex), max);
		return (max>=t.max);
	}

	BitSet toChoose;


	public void print(ArrayList<Clause> cs) {
		java.util.Iterator<Clause> i = cs.iterator();
		while(i.hasNext()) {
			Clause t = i.next();
			System.err.println(t);
		}
		System.err.println();
	}

	public void print(ArrayList<Clause> cs, int[] sel) {
		java.util.Iterator<Clause> i = cs.iterator();
		while(i.hasNext()) {
			Clause t = i.next();

			for (int a=0;a<m_numAttribs;a++) {
				if (a!=m_classIndex) {
					if (sel[a]==0) System.err.print(t.getVariableValue(a)+" ");
					else  System.err.print(sel[a]+" ");
				}
			}
			System.err.println("("+t.setValues+")");

		}
		System.err.println();
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
