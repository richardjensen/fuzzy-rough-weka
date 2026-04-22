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
import weka.core.Utils;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.measure.Measure;
import weka.fuzzy.measure.WeakGamma;
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
import java.util.Iterator;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * HillClimber :<br/>
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
public class HillClimber 
extends ASSearch 
implements RankedOutputSearch, StartSetHandler, OptionHandler {

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

	protected boolean m_prune=false;
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
	public HillClimber () {
		m_threshold = -Double.MAX_VALUE;
		m_doneRanking = false;
		m_prune=false;
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
		return "Hill Climber :\n\nPerforms a greedy forward or backward search "
				+"through "
				+"the space of attribute subsets. May start with no/all attributes or from "
				+"an arbitrary point in the space. Stops when the addition/deletion of any "
				+"remaining attributes results in a decrease in evaluation. "
				+"Can also produce a ranked list of "
				+"attributes by traversing the space from one side to the other and "
				+"recording the order that attributes are selected.\n";
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String searchBackwardsTipText() {
		return "Search backwards rather than forwards.";
	}

	public boolean getSuccessiveSearch() {
		return successiveSearch;
	}
	
	public void setSuccessiveSearch(boolean b) {
		successiveSearch = b;
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
	 * Set the threshold by which the AttributeSelection module can discard
	 * attributes.
	 * @param threshold the threshold.
	 */
	public void setThreshold(double threshold) {
		m_threshold = threshold;
	}

	/**
	 * Returns the threshold so that the AttributeSelection module can
	 * discard attributes from the ranking.
	 */
	public double getThreshold() {
		return m_threshold;
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


	public void setConsiderCorrelation(boolean c) {
		considerCorrelation = c;
	}

	public boolean getConsiderCorrelation() {
		return considerCorrelation;
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
		
		setSuccessiveSearch(Utils.getFlag('S', options));

		optionString = Utils.getOption('Z', options);
		if (optionString.length() != 0) {
			setStartSet(optionString);
		}
		
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

		setGenerateRanking(Utils.getFlag('R', options));
		setConsiderCorrelation(Utils.getFlag('D', options));
		setPrune(Utils.getFlag('P', options));

		optionString = Utils.getOption('T', options);
		if (optionString.length() != 0) {
			Double temp;
			temp = Double.valueOf(optionString);
			setThreshold(temp.doubleValue());
		}

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
		String[] options = new String[22];
		int current = 0;

		if (getSearchBackwards()) {
			options[current++] = "-B";
		}

		if (getConservativeForwardSelection()) {
			options[current++] = "-C";
		}
		
		if (getSuccessiveSearch()) {
			options[current++] = "-S";
		}

		if (!(getStartSet().equals(""))) {
			options[current++] = "-Z";
			options[current++] = ""+startSetToString();
		}
		
		options[current++] = "-Y";
		options[current++] = ""+(m_RedundancyMeasure.getClass().getName() + " " +
				Utils.joinOptions(m_RedundancyMeasure.getOptions())).trim();

		if (getGenerateRanking()) {
			options[current++] = "-R";
		}

		if (getConsiderCorrelation()) {
			options[current++] = "-D";
		}

		if (getPrune()) {
			options[current++] = "-P";
		}

		options[current++] = "-T";
		options[current++] = "" + getThreshold();

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

		if ((m_threshold != -Double.MAX_VALUE) && (m_doneRanking)) {
			FString.append("\tThreshold for discarding attributes: "
					+ Utils.doubleToString(m_threshold,8,4)+"\n");
		}

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
		long startTime = System.nanoTime();

		int i;
		double best_merit = -Double.MAX_VALUE;
		double temp_best,temp_merit;
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

		m_startRange.setUpper(m_numAttribs-1);
		if (!(getStartSet().equals(""))) {
			m_starting = m_startRange.getSelection();
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

		if (m_numToSelect<0) m_numToSelect=m_numAttribs;

		// Evaluate the initial subset
		best_merit = ASEvaluator.evaluateSubset(m_best_group);

		if (best_merit==1 && (m_ASEval instanceof FuzzyRoughSubsetEval) && !m_backward) {
			System.err.println(m_best_group+" => "+best_merit);
			return  attributeList(m_best_group);
		}

		// main search loop
		boolean done = false;
		boolean addone = false;
		boolean z;

		while (!done) {
			temp_group = (BitSet)m_best_group.clone();

			if (m_backward) {
				temp_best = alpha;				
			}
			else {
				//this enables search for non-monotonic measures
				//temp_best=-1;
				temp_best = best_merit;
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
					temp_merit = ASEvaluator.evaluateSubset(temp_group);
					if (m_backward) {
						z = (temp_merit >= temp_best);
					} else {
						if (considerCorrelation) {
							double discount = 0;

							if (m_best_group.cardinality()!=0) {

								for (int a = m_best_group.nextSetBit(0); a >= 0; a = m_best_group.nextSetBit(a+1)) {
									discount+= 0.5*(dependencies[i][a] + dependencies[a][i]);
								}

								discount/=m_best_group.cardinality();
							}

							temp_merit -= discount;
						}

						if (m_conservativeSelection) {
							z = (temp_merit >= temp_best);
						} else {
							z = (temp_merit > temp_best);
						}
					}

					if (z) {
						temp_best = temp_merit;
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
				double prev = best_merit;
				best_merit = temp_best;

				int val = (int)(100*(double)best_merit);

				if (m_backward) {
					m_best_group.clear(temp_index);
					searchOutput.append(m_best_group+" => "+best_merit+"\n");
					//System.err.println(m_best_group+" => "+best_merit);
				} else {
					if (considerCorrelation) {
						if (prev>best_merit) {
							done=true;
							break;
						}
					}
					
					m_best_group.set(temp_index);
					searchOutput.append(m_best_group+" => "+best_merit+"\n");
					//System.err.println(m_best_group+" => "+best_merit);

					//try removing redundant features
					if (m_prune) {		
						for (int a = m_best_group.nextSetBit(0); a >= 0; a = m_best_group.nextSetBit(a + 1)) {
							m_best_group.clear(a);

							double val2 = ASEvaluator.evaluateSubset(m_best_group);
							if (val2==best_merit&&val2!=0) { //prune this feature as it's redundant
								searchOutput.append("Pruned to: "+m_best_group+" => "+best_merit+"\n");
								System.err.println("Pruned to: "+m_best_group+" => "+best_merit);
							}
							else m_best_group.set(a);
						}

					}

					if (best_merit>=alpha) {
						done=true;
						break;
					}
					
					
				}

				if (m_best_group.cardinality()==m_numToSelect) {
					System.err.println("Stopping search - number of attributes limit reached");
					done=true;
					break;
				}

				m_rankedAtts[m_rankedSoFar][0] = temp_index;
				m_rankedAtts[m_rankedSoFar][1] = best_merit;
				m_rankedSoFar++;
			}
		}
		m_bestMerit = best_merit;
		
		long endTime = System.nanoTime();
		long durationNanos = endTime - startTime;
		double  m_searchTime = durationNanos / 1_000_000_000.0; // Convert nanoseconds to seconds
		
		// Optional: Print time immediately for quick feedback
		System.err.println("HillClimber search completed in: " + Utils.doubleToString(m_searchTime, 8, 4) + " seconds.");
		
		if (successiveSearch&&!m_backward) {
			ArrayList<BitSet> subsets = new ArrayList<BitSet>(5); //keeps track of all the groups
			done=false;
			BitSet ignore = (BitSet)m_best_group.clone();
			subsets.add(m_best_group);
			BitSet best_group = new BitSet(m_numAttribs);
			
			temp_best=-1;
			
			while (!done) {
				done=true;
				temp_group = (BitSet)best_group.clone();
				
				
				for (i=0;i<m_numAttribs;i++) {
					if (!ignore.get(i)&&i!=m_classIndex) {
						temp_group.set(i);
						temp_merit = ASEvaluator.evaluateSubset(temp_group);
						temp_group.clear(i);
						
						if (temp_merit>=temp_best) {
							temp_best = temp_merit;
							temp_index = i;
							done=false;
						}
					}
				}
				
				if (!done) {
					best_group.set(temp_index);
					ignore.set(temp_index);
				}
				else { //finished creating the group
					subsets.add((BitSet)best_group.clone());
					best_group.clear(); //reset the best group
					temp_best=-1;
					
					//if we haven't finished creating groups, loop again
					if (ignore.cardinality()!=m_numAttribs-1) done=false;
				}
			}
			
			String output="";
			String grouping="Groups are:\n";
			System.out.println("Groups are:");
			Iterator<BitSet> it = subsets.iterator();
			
			while (it.hasNext()) {
				BitSet group = it.next();
				
				grouping+="{";
				System.out.print("{");
				for (int a=0;a<group.length();a++) {
					if (group.get(a)) {
						output+=""+(a+1)+" ";
						grouping+=" "+m_Instances.attribute(a).name()+" ";
						System.out.print(" "+m_Instances.attribute(a).name()+" ");
					}
				}
				output+="; ";
				grouping+="}\n";
				System.out.println("}");
			}
			
			System.out.println(output);
			searchOutput.append(grouping);
		}

		/*
		//try removing features
		//(no point if we're already performing a backward search)
		if (m_prune&&!m_backward) {

			for (int a = m_best_group.nextSetBit(0); a >= 0; a = m_best_group.nextSetBit(a + 1)) {
					m_best_group.clear(a);

					double val = ASEvaluator.evaluateSubset(m_best_group);

					//if (val>=alpha && val<=tempVal) {
					if (val>=alpha) {
						m_bestMerit=val;
					}
					else m_best_group.set(a);
			}
			searchOutput.append("Pruned to: "+m_best_group+" => "+m_bestMerit+"\n");
			System.err.println("Pruned to: "+m_best_group+" => "+m_bestMerit);
		}*/

		return attributeList(m_best_group);
	}

	boolean successiveSearch=false;
	
	/**
	 * Produces a ranked list of attributes. Search must have been performed
	 * prior to calling this function. Search is called by this function to
	 * complete the traversal of the the search space. A list of
	 * attributes and merits are returned. The attributes a ranked by the
	 * order they are added to the subset during a forward selection search.
	 * Individual merit values reflect the merit associated with adding the
	 * corresponding attribute to the subset; because of this, merit values
	 * may initially increase but then decrease as the best subset is
	 * "passed by" on the way to the far side of the search space.
	 *
	 * @return an array of attribute indexes and associated merit values
	 * @throws Exception if something goes wrong.
	 */
	public double [][] rankedAttributes() throws Exception {

		if (m_rankedAtts == null || m_rankedSoFar == -1) {
			throw new Exception("Search must be performed before attributes "
					+"can be ranked.");
		}

		m_doRank = true;
		search (m_ASEval, null);

		double [][] final_rank = new double [m_rankedSoFar][2];
		for (int i=0;i<m_rankedSoFar;i++) {
			final_rank[i][0] = m_rankedAtts[i][0];
			final_rank[i][1] = m_rankedAtts[i][1];
		}

		resetOptions();
		m_doneRanking = true;

		if (m_numToSelect > final_rank.length) {
			throw new Exception("More attributes requested than exist in the data");
		}

		if (m_numToSelect <= 0) {
			if (m_threshold == -Double.MAX_VALUE) {
				m_calculatedNumToSelect = final_rank.length;
			} else {
				determineNumToSelectFromThreshold(final_rank);
			}
		}

		return final_rank;
	}

	private void determineNumToSelectFromThreshold(double [][] ranking) {
		int count = 0;
		for (int i = 0; i < ranking.length; i++) {
			if (ranking[i][1] > m_threshold) {
				count++;
			}
		}
		m_calculatedNumToSelect = count;
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
