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
 *    MultiReduct.java
 *   
 *
 */


import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.Utils;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.measure.Measure;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity3;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormLukasiewicz;
import weka.fuzzy.unsupervised.UWeakGamma;
import weka.fuzzy.unsupervised.UnsupervisedFuzzyMeasure;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * MultiReduct :<br/>
 * <br/>
 * Performs a greedy forward or backward search through the space of attribute subsets. May start with no/all attributes or from an arbitrary point in the space. 
 * Stops when the addition/deletion of any remaining attributes results in a decrease in evaluation. Can also produce a ranked list of attributes by traversing the space from one side to the other and recording the order that attributes are selected.<br/>
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
 *  <pre> -A
 *  The number of groups to form.</pre>
 <!-- options-end -->
 *
 * @author Richard Jensen
 * @version $Revision: 1.9 $
 */
public class MultiReduct 
extends ASSearch 
implements OptionHandler {

	/** for serialization */
	static final long serialVersionUID = -6312951970168525471L;

	/** does the data have a class */
	protected boolean m_hasClass;

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	protected boolean m_prune=false;

	/** the merit of the best subset found */
	protected double m_bestMerit;

	/** the best subset found */
	protected BitSet m_best_group;
	protected ASEvaluation m_ASEval;

	protected Instances m_Instances;

	/** Use a backwards search instead of a forwards one */
	protected boolean m_backward = false;

	/** If set then attributes will continue to be added during a forward
      search as long as the merit does not degrade */
	protected boolean m_conservativeSelection = false;

	protected int groups=1;

	public Similarity m_Similarity = new Similarity3();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity = new SimilarityEq();

	public TNorm m_TNorm = new TNormLukasiewicz();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();
	public UnsupervisedFuzzyMeasure m_RedundancyMeasure = new UWeakGamma();

	/**
	 * Constructor
	 */
	public MultiReduct () {
		m_prune=false;
		resetOptions();
	}

	/**
	 * Returns a string describing this search method
	 * @return a description of the search suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "MultiReduct :\n\nPerforms repeated greedy forward search.\n";
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
	}

	/**
	 * Get whether to search backwards
	 *
	 * @return true if the search will proceed backwards
	 */
	public boolean getSearchBackwards() {
		return m_backward;
	}


	public void setConsiderCorrelation(boolean c) {
		considerCorrelation = c;
	}

	public boolean getConsiderCorrelation() {
		return considerCorrelation;
	}

	//The number of groups to form
	public void setGroups(int a) {
		groups=a;
	}

	public int getGroups() {
		return groups;
	}

	public void setPrune(boolean p) {
		m_prune=p;
	}

	public boolean getPrune() {
		return m_prune;
	}

	public void setRedundancyMeasure(UnsupervisedFuzzyMeasure fe) {
		m_RedundancyMeasure = fe;
	}

	public UnsupervisedFuzzyMeasure getRedundancyMeasure() {
		return m_RedundancyMeasure;
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

		optionString = Utils.getOption('Y', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid Measure specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setRedundancyMeasure( (UnsupervisedFuzzyMeasure) Utils.forName( UnsupervisedFuzzyMeasure.class, className, moreOptions) );
		}
		else {
			setRedundancyMeasure(new UWeakGamma());
		}

		setConsiderCorrelation(Utils.getFlag('D', options));
		setPrune(Utils.getFlag('P', options));

		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setGroups(Integer.parseInt(optionString));
		}
	}

	/**
	 * Gets the current settings.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions () {
		String[] options = new String[16];
		int current = 0;

		if (getSearchBackwards()) {
			options[current++] = "-B";
		}

		options[current++] = "-Y";
		options[current++] = ""+(m_RedundancyMeasure.getClass().getName() + " " +
				Utils.joinOptions(m_RedundancyMeasure.getOptions())).trim();

		if (getConsiderCorrelation()) {
			options[current++] = "-D";
		}

		if (getPrune()) {
			options[current++] = "-P";
		}

		options[current++] = "-A";
		options[current++] = ""+getGroups();

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
		FString.append("\tMultiReduct ("
				+ ((m_backward)
						? "backwards)"
								: "forwards)")+".\n\tStart set: ");


		FString.append(searchOutput);

		return FString.toString();
	}

	private StringBuffer searchOutput;

	boolean considerCorrelation=false;

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
		double best_merit = -Double.MAX_VALUE;
		double temp_merit;
		int temp_index=0;
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

		if (!(m_ASEval instanceof SubsetEvaluator)) {
			throw  new Exception(m_ASEval.getClass().getName() 
					+ " is not a " 
					+ "Subset evaluator!");
		}


		if (m_ASEval instanceof UnsupervisedSubsetEvaluator) {
			m_hasClass = false;
			m_classIndex = -1;
		}
		else {
			m_hasClass = true;
			m_classIndex = m_Instances.classIndex();
		}

		SubsetEvaluator ASEvaluator = (SubsetEvaluator)m_ASEval;


		if (m_backward) {
			for (int a = 0; a < m_numAttribs; a++) {
				if (a != m_classIndex) {
					m_best_group.set(a);
				}
			}
		}

		double[][] dependencies = new double[m_numAttribs][m_numAttribs];

		if (!m_backward && (m_ASEval instanceof FuzzyRoughSubsetEval)) {
			FuzzyRoughSubsetEval ev = (FuzzyRoughSubsetEval)m_ASEval;
			if (ev.computeCore) {
				ev.core = ev.computeCore();
				m_best_group = (BitSet)ev.core.clone();
			}

			if (considerCorrelation) {
				boolean m_isNumeric = m_Instances.attribute(m_classIndex).isNumeric();

				if (m_isNumeric) {
					m_DecisionSimilarity = m_Similarity;
				}
				else m_DecisionSimilarity = m_SimilarityEq;

				m_Similarity.setInstances(m_Instances);
				m_DecisionSimilarity.setInstances(m_Instances);

				m_RedundancyMeasure.set(m_Similarity,m_DecisionSimilarity,m_TNorm,m_Similarity.getTNorm(),m_Implicator,m_SNorm,m_Instances.size(),m_numAttribs,m_classIndex,m_Instances);

				for (int a=0;a<m_numAttribs;a++) {
					if (a!=m_classIndex) {
						BitSet temp = new BitSet(m_numAttribs);
						temp.set(a);

						for (int b=0;b<m_numAttribs;b++) {
							if (b!=m_classIndex) {
								if (a==b) dependencies[a][a] = 1;
								else dependencies[a][b] = m_RedundancyMeasure.calculate(temp, b);
							}
						}
					}
				}
			}
		}
		else considerCorrelation = false;

		// Evaluate the initial subset
		best_merit = ASEvaluator.evaluateSubset(m_best_group);

		if (best_merit==1 && (m_ASEval instanceof FuzzyRoughSubsetEval) && !m_backward) {
			System.err.println(m_best_group+" => "+best_merit);
			return  attributeList(m_best_group);
		}

		// main search loop
		boolean done = false;

		done=false;
		int numGroups = (int)groups;

		BitSet ignore = new BitSet(m_numAttribs);
		BitSet[] best_groups = new BitSet[numGroups];

		//combine pairs of features - no group information needed
		if (pairwise) {
			ArrayList<BitSet> groups = new ArrayList<BitSet>(m_numAttribs);
			
			//initialise
			for (int a=0;a<m_numAttribs;a++) {
				BitSet toAdd = new BitSet(m_numAttribs);
				toAdd.set(a);
				groups.add(toAdd);
			}
			
			
		}
		else { // need number of groups to be specified
			//initialise
			for (int a=0;a<numGroups;a++) best_groups[a] = new BitSet(m_numAttribs);
			double[] bestSoFar = new double[numGroups];
			for (int g=0;g<numGroups;g++) bestSoFar[g]=-1;
			boolean[] groupFinished = new boolean[numGroups];

			while (!done) {
				done=true;
				//for each group, perform the next addition of features to the group
				for (int g=0;g<numGroups;g++) {
					temp_group = (BitSet)best_groups[g].clone();
					if (!groupFinished[g]) {
						groupFinished[g] = true;

						for (int a=0;a<m_numAttribs;a++) {
							if (!ignore.get(a)&&a!=m_classIndex) {
								temp_group.set(a);
								temp_merit = ASEvaluator.evaluateSubset(temp_group);
								temp_group.clear(a);

								if (temp_merit>bestSoFar[g]) {
									bestSoFar[g] = temp_merit;
									temp_index = a;
									groupFinished[g]=false;
								}
							}
						}

						if (!groupFinished[g]) {
							best_groups[g].set(temp_index);
							ignore.set(temp_index);
						}
					}
					//System.err.println(g+": "+best_groups[g]+" "+bestSoFar[g]+" "+groupFinished[g]);
				}

				boolean alldone=true;
				for (int g=0;g<numGroups;g++) {
					if (!groupFinished[g]) alldone=false;
				}

				if (ignore.cardinality()!=m_numAttribs-1) done=false;
				if (alldone) done=true;
			}
		}
		
		String[] granularities = null;
		if (outputGranularity&&(m_ASEval instanceof FuzzyRoughSubsetEval)) {
			FuzzyRoughSubsetEval ev = (FuzzyRoughSubsetEval)m_ASEval;
			granularities = ev.outputGranularities();
						
		}

		String output="";
		String grouping="Groups are:\n";
		System.out.println("Groups are:");
		String partitions="";
		
		for (int g=0;g<numGroups;g++) {
			BitSet group = best_groups[g];

			grouping+="{";
			System.out.print("{");
			for (int a=0;a<group.length();a++) {
				if (group.get(a)) {
					output+=""+(a+1)+" ";
					grouping+=" "+m_Instances.attribute(a).name()+" ";
					System.out.print(" "+m_Instances.attribute(a).name()+" ");
					if (outputGranularity&&(m_ASEval instanceof FuzzyRoughSubsetEval)) partitions+=granularities[a]+" ";
				}
			}
			output+="; ";
			grouping+="}\n";
			System.out.println("}");
			if (outputGranularity&&(m_ASEval instanceof FuzzyRoughSubsetEval)) partitions+=";";
		}
		
		System.out.println(output);
		searchOutput.append(grouping);
		if (outputGranularity&&(m_ASEval instanceof FuzzyRoughSubsetEval)) {
			System.out.println(partitions);
			searchOutput.append(partitions);
		}
		
		return attributeList(m_best_group);
	}

	boolean outputGranularity=true;
	boolean successiveSearch=false;
	boolean pairwise=false;

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
	}
}
