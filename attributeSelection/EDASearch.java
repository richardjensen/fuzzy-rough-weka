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
 *    EDASearch.java
 *    Copyright (C) Richard Jensen
 *
 */

package  weka.attributeSelection;

import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;

import java.text.SimpleDateFormat;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Calendar;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Random;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * EDASearch:<br/>
 * <br/>
 * Performs a search using an Estimation of Distribution Algorithm.<br/>
 * <br/>
 *
 * @author Richard Jensen
 */
public class EDASearch 
extends ASSearch 
implements OptionHandler, TechnicalInformationHandler {

	public int dummy=1; //to get around Weka Experimenter problems
	/** for serialization */
	static final long serialVersionUID = -1618264232838472679L;

	/** holds the class index */
	private int m_classIndex;

	/** number of attributes in the data */
	private int m_numAttribs;

	/** the current population */
	//private EDABitSet [] m_population;

	/** the number of individual solutions */
	private int m_numIndividuals;

	/** the probability vector for each individual for each population**/
	private double[][] m_probVec;

	private EDABitSet m_gBest;
	private double gBest_merit;
	private int gBest_cardinality;

	/** the number of entries to cache for lookup */
	protected int m_lookupTableSize;

	/** the lookup table */
	private Hashtable<Integer, Double> m_lookupTable;

	/** random number generation */
	private Random m_random;

	/** seed for random number generation */
	private int m_seed;

	static final int CGA=0;
	static final int RKJ=1;
	static final int PBIL=2;
	static final int PBIL2=3;

	public static final Tag[] TAGS_EVALUATION = {
		new Tag(CGA, "Compact GA"),
		new Tag(RKJ, "Reducts only"),
		new Tag(PBIL, "Population-based Incremental Learning"),
		new Tag(PBIL2, "Population-based Incremental Learning v 2")
	};

	private int STRATEGY = RKJ;

	private int m_numberOfPopulations=1;

	private double m_learnRate=0.1;
	private double m_mutationRate = 0.02;
	private double m_mutationShift = 0.05;
	private double m_negLearnRate=2;

	protected boolean m_verbose=false;
	protected boolean m_prune=false;

	/** the maximum number of generations to evaluate */
	private int m_maxGenerations;

	/** how often reports are generated */
	private int m_reportFrequency;

	/** holds the generation reports */
	private StringBuffer m_generationReports;

	double subsetAlpha = 0;


	// Inner class
	/**
	 * A bitset for the genetic algorithm
	 */
	protected class EDABitSet 
	implements Cloneable, Serializable, RevisionHandler {

		/** for serialization */
		static final long serialVersionUID = -2930607837482622224L;

		/** the bitset */
		private BitSet m_chromosome;

		/** holds raw merit */
		private double m_objective = -Double.MAX_VALUE;

		/** the fitness */
		private double m_fitness;

		/**
		 * Constructor
		 */
		public EDABitSet () {
			m_chromosome = new BitSet();
		}

		/**
		 * Constructor
		 */
		public EDABitSet (int a) {
			m_chromosome = new BitSet(a);
		}

		/**
		 * makes a copy of this GABitSet
		 * @return a copy of the object
		 * @throws CloneNotSupportedException if something goes wrong
		 */
		public Object clone() throws CloneNotSupportedException {
			EDABitSet temp = new EDABitSet();

			temp.setObjective(this.getObjective());
			temp.setFitness(this.getFitness());
			temp.setSubset((BitSet)(this.m_chromosome.clone()));
			return temp;
			//return super.clone();
		}

		/**
		 * sets the objective merit value
		 * @param objective the objective value of this population member
		 */
		public void setObjective(double objective) {
			m_objective = objective;
		}

		/**
		 * gets the objective merit
		 * @return the objective merit of this population member
		 */
		public double getObjective() {
			return m_objective;
		}

		/**
		 * sets the scaled fitness
		 * @param fitness the scaled fitness of this population member
		 */
		public void setFitness(double fitness) {
			m_fitness = fitness;
		}

		/**
		 * gets the scaled fitness
		 * @return the scaled fitness of this population member
		 */
		public double getFitness() {
			return m_fitness;
		}

		/**
		 * get the chromosome
		 * @return the chromosome of this population member
		 */
		public BitSet getSubset() {
			return m_chromosome;
		}

		/**
		 * set the chromosome
		 * @param c the chromosome to be set for this population member
		 */
		public void setSubset(BitSet c) {
			m_chromosome = c;
		}

		/**
		 * unset a bit in the chromosome
		 * @param bit the bit to be cleared
		 */
		public void clear(int bit) {
			m_chromosome.clear(bit);
		}

		/**
		 * set a bit in the chromosome
		 * @param bit the bit to be set
		 */
		public void set(int bit) {
			m_chromosome.set(bit);
		}

		/**
		 * get the value of a bit in the chromosome
		 * @param bit the bit to query
		 * @return the value of the bit
		 */
		public boolean get(int bit) {
			return m_chromosome.get(bit);
		}

		/**
		 * Returns the revision string.
		 * 
		 * @return		the revision
		 */
		public String getRevision() {
			return RevisionUtils.extract("$Revision: 5286 $");
		}
	}

	/**
	 * Returns an enumeration describing the available options.
	 * @return an enumeration of all the available options.
	 **/
	public Enumeration<Option> listOptions () {
		Vector<Option> newVector = new Vector<Option>(6);

		newVector.addElement(new Option(
				"\tclass name of the subset evaluator to use for the heuristic desirability of edges.\n"
						+ "\tPlace any classifier options LAST on the command line\n"
						+ "\tfollowing a \"--\".\n"
						+ "\t(default: weka.attributeSelection.FuzzyRoughSubsetEval)", 
						"L", 1, "-L <subset evaluator>"));
		newVector.addElement(new Option("\tSet the size of the population."
				+"\n\t(default = 10)."
				, "Z", 1
				, "-Z <population size>"));
		newVector.addElement(new Option("\tSet the number of generations."
				+"\n\t(default = 20)" 
				, "G", 1, "-G <number of generations>"));
		newVector.addElement(new Option("\tSet the value of alpha."
				+"\n\t(default = 1)" 
				, "A", 1, "-A <alpha"
						+" value>"));    
		newVector.addElement(new Option("\tSet the value of beta."
				+"\n\t(default = 2)" 
				, "B", 1, "-B <beta value>"));

		newVector.addElement(new Option("\tSet frequency of generation reports."
				+"\n\te.g, setting the value to 5 will "
				+"\n\treport every 5th generation"
				+"\n\t(default = number of generations)" 
				, "R", 1, "-R <report frequency>"));
		newVector.addElement(new Option("\tSet the random number seed."
				+"\n\t(default = 1)" 
				, "S", 1, "-S <seed>"));

		return  newVector.elements();
	}

	/**
	 * Parses a given list of options. <p/>
	 *
	 *
	 * @param options the list of options as an array of strings
	 * @throws Exception if an option is not supported
	 *
	 **/
	public void setOptions (String[] options)
			throws Exception {
		String optionString;
		resetOptions();

		optionString = Utils.getOption('Z', options);
		if (optionString.length() != 0) {
			setNumIndividuals(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('G', options);
		if (optionString.length() != 0) {
			setMaxGenerations(Integer.parseInt(optionString));
			setReportFrequency(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setLearnRate((new Double(optionString)).doubleValue());
		}

		optionString = Utils.getOption('B', options);
		if (optionString.length() != 0) {
			setNegLearnRate((new Double(optionString)).doubleValue());
		}

		optionString = Utils.getOption('R', options);
		if (optionString.length() != 0) {
			setReportFrequency(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('S', options);
		if (optionString.length() != 0) {
			setSeed(Integer.parseInt(optionString));
		}
		
		optionString = Utils.getOption('C', options);
		if (optionString.length() != 0) {
			setLookupTableSize(Integer.parseInt(optionString));
		}

		setPrune(Utils.getFlag('P', options));

		setVerbose(Utils.getFlag('V', options));

		optionString = Utils.getOption('Q', options);
		if (optionString.length() != 0) {
			setNumberOfPopulations((new Integer(optionString)).intValue());
		}


		optionString = Utils.getOption('M', options);
		if (optionString.length() != 0) {
			STRATEGY = Integer.parseInt(optionString);
		}
	}

	public void setLookupTableSize(int n) {
		if (n<0) n=10;
		m_lookupTableSize=n;
	}
	
	public int getLookupTableSize() {
		return m_lookupTableSize;
	}

	public void setNumberOfPopulations(int n) {
		m_numberOfPopulations=n;
	}

	public int getNumberOfPopulations() {
		return m_numberOfPopulations;
	}

	public SelectedTag getApproach() {
		return new SelectedTag(STRATEGY, TAGS_EVALUATION);
	}

	/**
	 * Sets the performance evaluation measure to use for selecting attributes
	 * for the decision table
	 * 
	 * @param newMethod the new performance evaluation metric to use
	 */
	public void setApproach(SelectedTag newMethod) {
		if (newMethod.getTags() == TAGS_EVALUATION) {
			STRATEGY = newMethod.getSelectedTag().getID();
		}
	}

	/**
	 * Gets the current settings.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions () {
		String[] heuristicOptions = new String[0];

		String[] options = new String[22 + heuristicOptions.length+1];
		int current = 0;

		options[current++] = "-M";
		options[current++] = "" + STRATEGY;

		options[current++] = "-Z";
		options[current++] = "" + getNumIndividuals();
		options[current++] = "-G";
		options[current++] = "" + getMaxGenerations();
		options[current++] = "-A";
		options[current++] = "" + getLearnRate();
		options[current++] = "-B";
		options[current++] = "" + getNegLearnRate();
		options[current++] = "-R";
		options[current++] = "" + getReportFrequency();
		options[current++] = "-S";
		options[current++] = "" + getSeed();
		options[current++] = "-C";
		options[current++] = "" + getLookupTableSize();
		options[current++] = "-Q";
		options[current++] = "" + getNumberOfPopulations();

		if (getPrune()) {
			options[current++] = "-P";
		}

		if (getVerbose()) {
			options[current++] = "-V";
		}

		if (heuristicOptions.length > 0) {
			options[current++] = "--";
			System.arraycopy(heuristicOptions, 0, options, current, heuristicOptions.length);
			current += heuristicOptions.length;
		}

		while (current < options.length) {
			options[current++] = "";
		}
		return  options;
	}

	public String pruneTipText() {
		return "Prune the subset after search to remove redundant features";
	}

	public void setPrune(boolean p) {
		m_prune=p;
	}

	public boolean getPrune() {
		return m_prune;
	}

	public boolean getVerbose() {
		return m_verbose;
	}

	public void setVerbose(boolean v) {
		m_verbose=v;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String seedTipText() {
		return "Set the random seed.";
	}

	/**
	 * set the seed for random number generation
	 * @param s seed value
	 */
	public void setSeed(int s) {
		m_seed = s;
	}

	/**
	 * get the value of the random number generator's seed
	 * @return the seed for random number generation
	 */
	public int getSeed() {
		return m_seed;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String reportFrequencyTipText() {
		return "Set how frequently reports are generated. Default is equal to "
				+"the number of generations meaning that a report will be printed for "
				+"initial and final generations. Setting the value to 5 will result in "
				+"a report being printed every 5 generations.";
	}

	/**
	 * set how often reports are generated
	 * @param f generate reports every f generations
	 */
	public void setReportFrequency(int f) {
		m_reportFrequency = f;
	}

	/**
	 * get how often reports are generated
	 * @return how often reports are generated
	 */
	public int getReportFrequency() {
		return m_reportFrequency;
	}


	/**
	 * get learning rate
	 * @return m_learnRate, the acceleration constant 
	 */
	public double getLearnRate() {
		return m_learnRate;
	}

	/**
	 * set the learning rate
	 * @param a, 
	 */
	public void setLearnRate(double a) {
		m_learnRate = a;
	}

	/**
	 * get negative learning rate (used in PBIL v2)
	 * @return m_negLearnRate, acceleration constant
	 */
	public double getNegLearnRate() {
		return m_negLearnRate;
	}

	/**
	 * set the negative learning rate (used in PBIL v2)
	 * @param b, 
	 */
	public void setNegLearnRate(double b) {
		m_negLearnRate = b;
	}

	/**
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String maxGenerationsTipText() {
		return "Set the number of generations to evaluate.";
	}

	/**
	 * set the number of generations to evaluate
	 * @param m the number of generations
	 */
	public void setMaxGenerations(int m) {
		m_maxGenerations = m;
	}

	/**
	 * get the number of generations
	 * @return the maximum number of generations
	 */
	public int getMaxGenerations() {
		return m_maxGenerations;
	}


	/**
	 * set the population size
	 * @param p the size of the population
	 */
	public void setNumIndividuals(int p) {
		m_numIndividuals = p;
	}

	/**
	 * get the size of the population
	 * @return the population size
	 */
	public int getNumIndividuals() {
		return m_numIndividuals;
	}

	/**
	 * Returns a string describing this search method
	 * @return a description of the search suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return "EDASearch:\n\nPerforms a search using an Estimation of Distribution Algorithm. Setting the learning rate to 0 will make the program estimate this value. A learning rate of around 0.1 is recommended for PBIL. Multiple populations can be used (the program will divide the original population into m subpopulations).\n\n";
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
		result.setValue(Field.AUTHOR, "X. Wang, J. Yang, X. Teng, W. Xia, and R. Jensen");
		result.setValue(Field.YEAR, "2007");
		result.setValue(Field.TITLE, "Feature Selection based on Rough Sets and Particle Swarm Optimization");
		result.setValue(Field.JOURNAL, "Pattern Recognition Letters");
		result.setValue(Field.PAGES, "459-471");
		result.setValue(Field.VOLUME, "28");
		result.setValue(Field.NUMBER, "4");

		additional = result.add(Type.ARTICLE);
		additional.setValue(Field.AUTHOR, "X. Wang, J. Yang, R. Jensen and X. Liu");
		additional.setValue(Field.TITLE, "Rough Set Feature Selection and Rule Induction for Prediction of Malignancy Degree in Brain Glioma");
		additional.setValue(Field.BOOKTITLE, "2003 UK Workshop on Computational Intelligence");
		additional.setValue(Field.YEAR, "2006");
		result.setValue(Field.JOURNAL, "Computer Methods and Programs in Biomedicine");
		additional.setValue(Field.PAGES, "147-156");
		result.setValue(Field.VOLUME, "83");
		result.setValue(Field.NUMBER, "2");

		return result;
	}

	/**
	 * Constructor. Make a new EDASearch object
	 */
	public EDASearch() {
		resetOptions();
		m_random = new Random(m_seed);
		dummy = m_random.nextInt();
	}

	/**
	 * returns a description of the search
	 * @return a description of the search as a String
	 */
	public String toString() {
		StringBuffer desc = new StringBuffer();
		desc.append("\nEDA search: "+TAGS_EVALUATION[STRATEGY].getReadable()+"\n");
		desc.append("\tPopulation size: "+m_numIndividuals);
		desc.append("\tPopulations: "+m_numberOfPopulations);
		desc.append("\n\tNumber of generations: "+m_maxGenerations);
		desc.append("\n\tLearning rate: "
				+Utils.doubleToString(m_learnRate,6,3));
		desc.append("\n\tNegative learning rate (PBIL v2 only): "
				+Utils.doubleToString(m_negLearnRate,6,3));
		desc.append("\n\tReport frequency: "+m_reportFrequency);
		desc.append("\n\tRandom number seed: "+m_seed+"\n");
		desc.append(m_generationReports.toString());
		return desc.toString();
	}

	/**
	 * Searches the attribute subset space using a PSO algorithm.
	 *
	 * @param ASEval the attribute evaluator to guide the search
	 * @param data the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception if the search can't be completed
	 */
	public int[] search (ASEvaluation ASEval, Instances data)
			throws Exception {

		m_generationReports = new StringBuffer();

		if (!(ASEval instanceof SubsetEvaluator)) {
			throw  new Exception(ASEval.getClass().getName() 
					+ " is not a " 
					+ "Subset evaluator!");
		}


		if (ASEval instanceof UnsupervisedSubsetEvaluator) {
		}
		else {
			m_classIndex = data.classIndex();
		}

		SubsetEvaluator ASEvaluator = (SubsetEvaluator)ASEval;
		m_numAttribs = data.numAttributes();

		int extra = (int)((double)m_lookupTableSize*0.25);
		
		// initial random population
		m_lookupTable = new Hashtable<Integer, Double>(m_lookupTableSize+extra);
		keys = new ArrayList<Integer>(m_lookupTableSize);
		
		m_random = new Random(m_seed);

		//set up the global best subset
		m_gBest = new EDABitSet(m_numAttribs);
		gBest_merit = -999999999;
		gBest_cardinality=999999999;


		BitSet core=new BitSet(m_numAttribs);

		if (ASEvaluator instanceof FuzzyRoughSubsetEval) {
			FuzzyRoughSubsetEval ev = (FuzzyRoughSubsetEval)ASEvaluator;
			if (ev.computeCore) {
				core = ev.computeCore();
				ev.core = core;
			}
		}

		//estimate the learning rate if an invalid rate is chosen
		if (m_learnRate<=0) m_learnRate = Math.min(1, 1/Math.log10(m_numAttribs));
		System.err.println("Learning rate: "+m_learnRate);

		System.err.println(TAGS_EVALUATION[STRATEGY].getReadable());

		int numPBILs= m_numberOfPopulations;

		//Only one population allowed for this method
		if (STRATEGY==CGA) numPBILs=1;

		int numIndividuals = (int)(m_numIndividuals/numPBILs); //split the population equally

		//initially, the probabilities are 0.5
		m_probVec = new double[numPBILs][m_numAttribs];
		for (int j=0;j<numPBILs;j++) Arrays.fill(m_probVec[j], 0.5);

		EDABitSet [][] m_population = new EDABitSet[numPBILs][numIndividuals];

		String time = new SimpleDateFormat("d MMM yyyy HH:mm:ss").format(Calendar.getInstance().getTime());
		if (m_verbose) System.out.println("Started at: "+ time);
		//report 10 times on the estimated time to completion
		int factor = (int)((double)m_maxGenerations/10d);

		for (int i=1;i<=m_maxGenerations;i++) {
			long startTime = System.currentTimeMillis();

			//initialise the population
			for (int j=0;j<numPBILs;j++) {
				m_population[j] = initPopulation(ASEvaluator,core,j);		

				// evaluate each individual
				for (int p = 0; p < numIndividuals; p++) {
					evaluateIndividual(ASEvaluator,m_population[j][p]);
				}
			}
			switch (STRATEGY) {
			case CGA:
				updateProbabilitiesCGA(m_population,m_probVec,numIndividuals,numPBILs);
				break;
			case RKJ:
				updateProbabilitiesRKJ(m_population,m_probVec,numIndividuals,numPBILs);
				break;
			case PBIL: 
				updateProbabilitiesPBIL(m_population,m_probVec,numIndividuals,numPBILs);
				break;
			case PBIL2: 
				updateProbabilitiesPBIL2(m_population,m_probVec,numIndividuals,numPBILs);
				break;
			}

			if ((i == m_maxGenerations) || 
					((i % m_reportFrequency) == 0)) {
				m_generationReports.append(populationReport(i,m_population[0]));

			}

			long endTime = System.currentTimeMillis();
			long duration = endTime - startTime;

			if (m_verbose && (i==1||i%factor==0)) {
				Calendar cal = Calendar.getInstance();
				String timeStamp = new SimpleDateFormat("d MMM yyyy HH:mm:ss").format(cal.getTime());
				
				cal.add(Calendar.SECOND, (int)(duration*(m_maxGenerations-i)/1000));
				String timeStamp2 = new SimpleDateFormat("d MMM yyyy HH:mm:ss").format(cal.getTime());
				System.out.println("["+timeStamp+"]: ~"+(duration*((int)(m_maxGenerations-i))/60000)+" min(s) to go, finish at ["+timeStamp2+"]");
			}
		}

		System.err.println("Subset evaluations: "+evals+"\n");

		BitSet m_best_group = m_gBest.getSubset();
		double m_bestMerit = m_gBest.getObjective();

		//need a check here to make sure that a reduct is found, if not return the full set of features
		if (m_bestMerit < 1) System.err.println("*** Reduct not found! ***");  

		//Perform simple pruning - try removing features
		if (m_prune) {
			for (int a = m_best_group.nextSetBit(0); a >= 0; a = m_best_group.nextSetBit(a + 1)) {
				m_best_group.clear(a);			
				double val = ASEvaluator.evaluateSubset(m_best_group);

				//if (val>=alpha) {
				if (val>=m_bestMerit) {
					m_bestMerit=val;
				}
				else m_best_group.set(a);
			}
			m_generationReports.append("\nPruned to: "+m_best_group+" => "+m_bestMerit+"\n");
			System.err.println("Pruned to: "+m_best_group+" => "+m_bestMerit);
		}

		//print out the probabilities in the probVec array
		if (m_verbose) {
			for (int j=0;j<numPBILs;j++) {
				m_generationReports.append("\nPopulation "+(j+1)+", final attribute probabilities: \n");
				for (int a=0;a<m_numAttribs;a++) {
					if (a!=m_classIndex) {
						m_generationReports.append("Attr: "+(a+1)+" => "+m_probVec[j][a]+"\n");
					}
				}
			}
		}

		m_lookupTable.clear();
		
		System.err.println("Size of reduct: "+m_best_group.cardinality()+"\n");
		return attributeList(m_best_group);
	}

	private int evals=0;

	//Compact GA
	private void updateProbabilitiesCGA(EDABitSet[][] population, double[][] probVec, int numIndividuals, int numPBILs) {
		int best = -1;
		int bestCard=100000000;
		double bestObj=-1;
		int j=0; //assume only one population for this

		for (int a=0;a<numIndividuals;a++) {
			BitSet subset = population[j][a].getSubset();

			int cardinality = subset.cardinality();
			double obj =  population[j][a].getObjective();
			if (cardinality<1) {cardinality=1; obj=0.01;}

			if (bestObj<obj) {
				obj  = bestObj;
				best = a;
				bestCard = cardinality;
			}
			else if (bestObj==obj) {
				if (cardinality<bestCard) {
					best = a;
					bestCard = cardinality;
				}
			}

		}

		if (best==-1) System.err.println("best = -1");
		BitSet bestSubset = population[j][best].getSubset();

		//add the merit to the total for each attribute appearing in this subset
		for (int a = 0; a<m_numAttribs;a++) {	
			if (a!=m_classIndex) {
				if (bestSubset.get(a)) {
					probVec[j][a]+= m_learnRate/(float)population[j].length;

					if (probVec[j][a]>1) probVec[j][a]=1;

				}		

				else {
					probVec[j][a]-= m_learnRate/(float)population[j].length;
					//avoid zero probabilities 
					if (probVec[j][a]<=0)  probVec[j][a]=0.05;

				}
				//if (m_random.nextDouble() < m_mutationRate) m_probVec[j] = m_probVec[j] * (1d - m_mutationShift) + (m_random.nextBoolean() ? 1d : 0d) * m_mutationShift;
			}
		}

	}

	/** currently unused **/
	protected boolean useReducts = true; //only use subsets that are reducts to update the probabilities

	//Only update using reducts
	private void updateProbabilitiesRKJ(EDABitSet[][] population, double[][] probVec, int numIndividuals, int numPBILs) {
		// Find best subset
		BitSet[] bestSubset = new BitSet[numPBILs];
		double maxEval = -999999999;

		//for each population, find the best subset
		for (int j=0;j<numPBILs;j++) {
			maxEval = -999999999;
			double bestObjective = -9999999;

			for (int p=0;p<numIndividuals;p++) {
				double objective = population[j][p].getObjective();
				double eval = objective/population[j][p].getSubset().cardinality();

				// !!!
				//only consider proper reducts (i.e. don't consider the subset size here)
				if (eval > maxEval) {
					if (objective>=bestObjective) {
						bestObjective = objective;
						maxEval = eval;
						bestSubset[j] = population[j][p].getSubset();						
					}
				}

			}
		}

		// Update the probability vector, but attempt to keep the populations searching different areas
		for (int j=0;j<numPBILs;j++) {
			for (int a = 0; a < m_numAttribs; a++) {
				if (a!=m_classIndex) {
					probVec[j][a] = probVec[j][a] * (1d - m_learnRate) + (bestSubset[j].get(a) ? 1d : 0d) * m_learnRate;

					//mutation
					if (m_random.nextDouble() < m_mutationRate) {
						probVec[j][a] = probVec[j][a] * (1d - m_mutationShift) + (m_random.nextBoolean() ? 1d : 0d) * m_mutationShift;
					}
				}
			}
		}
	}

	//PBIL: Population-based Incremental Learning
	//ignores the negative learning rate
	private void updateProbabilitiesPBIL(EDABitSet[][] population, double[][] probVec, int numIndividuals, int numPBILs) {
		// Find best subset
		BitSet[] bestSubset = new BitSet[numPBILs];
		double maxEval = -999999999;

		//for each population
		for (int j=0;j<numPBILs;j++) {
			maxEval = -999999999;

			for (int p=0;p<numIndividuals;p++) {
				double eval = population[j][p].getObjective()/population[j][p].getSubset().cardinality();

				if (eval > maxEval) {
					maxEval = eval;
					bestSubset[j] = population[j][p].getSubset();
				}

			}
		}

		// Update the probability vector
		for (int j=0;j<numPBILs;j++) {
			for (int a = 0; a < m_numAttribs; a++) {
				if (a!=m_classIndex) {
					probVec[j][a] = probVec[j][a] * (1d - m_learnRate) + (bestSubset[j].get(a) ? 1d : 0d) * m_learnRate;

					//mutation
					if (m_random.nextDouble() < m_mutationRate) {
						probVec[j][a] = probVec[j][a] * (1d - m_mutationShift) + (m_random.nextBoolean() ? 1d : 0d) * m_mutationShift;
					}
				}
			}
		}
	}


	//PBIL2
	//Find the top two subsets and update accordingly
	private void updateProbabilitiesPBIL2(EDABitSet[][] population, double[][] probVec, int numIndividuals, int numPBILs) {
		// Find best subset
		BitSet[] bestSubset = new BitSet[numPBILs];
		BitSet[] bestSubset2 = new BitSet[numPBILs];
		double maxEval = -999999999, maxEval2=-999999999;

		//for each population
		for (int j=0;j<numPBILs;j++) {
			maxEval = -999999999; maxEval2 = -999999999;

			for (int p=0;p<numIndividuals;p++) {
				double eval = population[j][p].getObjective()/population[j][p].getSubset().cardinality();

				if (eval > maxEval2) {
					if (eval>maxEval) {
						maxEval2 = maxEval;
						maxEval = eval;
						bestSubset2 = bestSubset; //replace the previous best with the new one
						bestSubset[j] = population[j][p].getSubset();

					}
					else {
						maxEval2 = eval;
						bestSubset2[j] = population[j][p].getSubset();
					}
				}

			}
		}

		// Update the probability vector
		for (int j=0;j<numPBILs;j++) {
			for (int a = 0; a < m_numAttribs; a++) {
				if (a!=m_classIndex) {
					probVec[j][a] = probVec[j][a] * (1d - m_learnRate) + (bestSubset[j].get(a) ? 1d : 0d) * m_learnRate + (bestSubset2[j].get(a) ? 1d : 0d) * m_learnRate;

					//mutation
					if (m_random.nextDouble() < m_mutationRate) {
						probVec[j][a] = probVec[j][a] * (1d - m_mutationShift) + (m_random.nextBoolean() ? 1d : 0d) * m_mutationShift;
					}
				}
			}
		}

	}

	/**
	 * converts a BitSet into a list of attribute indexes 
	 * @param group the BitSet to convert
	 * @return an array of attribute indexes
	 **/
	private int[] attributeList (BitSet group) {
		int count = 0;

		// count how many were selected
		for (int i = 0; i < m_numAttribs; i++) {
			if (group.get(i)&&i!=m_classIndex) {
				count++;
			}
		}

		int[] list = new int[count];
		count = 0;

		for (int i = 0; i < m_numAttribs; i++) {
			if (group.get(i)&&i!=m_classIndex) {
				list[count++] = i;				
			}
		}

		return  list;
	}

	ArrayList<Integer> keys;

	/**
	 * evaluates a single particle. Population members are looked up in
	 * a hash table and if they are not found then they are evaluated using
	 * ASEvaluator.
	 * @param ASEvaluator the subset evaluator to use for evaluating population
	 * members
	 * @param particle, the particle to evaluate
	 * @throws Exception if something goes wrong during evaluation
	 */
	private double evaluateIndividual (SubsetEvaluator ASEvaluator, EDABitSet subset)
			throws Exception {
		double merit;
		boolean reachedCapacity = (keys.size()>=m_lookupTableSize);
		int hashCode = subset.getSubset().hashCode();
				
		// if its not in the lookup table then evaluate and insert (if space)
		if (!m_lookupTable.containsKey(hashCode)) {
			merit = ASEvaluator.evaluateSubset(subset.getSubset());
			evals++;

			if (merit>1) merit=1;
			subset.setObjective(merit);
			
			//if the hashtable is full, replace a random element
			//The idea is that newer subsets are more likely to be generated again
			//We maintain an ArrayList of keys as this allows random selection of keys, which is not possible with the Hashtable data structure on its own
			if (reachedCapacity) {
				int toRemove = (int)(m_random.nextDouble()*(double)keys.size());
				
				//get the key this index corresponds to
				Integer remove = keys.get(toRemove);
				m_lookupTable.remove(remove);
				keys.remove(toRemove);
			}
			
			//add the subset's hashcode to the hashtable and set of keys
			m_lookupTable.put(hashCode,merit);
			keys.add(hashCode);
			
			if ((merit > gBest_merit)||(merit==gBest_merit && subset.getSubset().cardinality() < gBest_cardinality)) {
				m_gBest = subset;
				gBest_merit = merit;
				gBest_cardinality = subset.getSubset().cardinality();
			}
		} else {
			merit = m_lookupTable.get(hashCode);
			subset.setObjective(merit);			
		}

		return merit;
	}

	/**
	 * creates random population members for the initial population.
	 * 
	 * @throws Exception if the population can't be created
	 */
	private EDABitSet[] initPopulation (SubsetEvaluator ASEvaluator, BitSet core, int j) throws Exception {
		int i;
		int start = 0;
		EDABitSet [] m_population = new EDABitSet [m_numIndividuals];

		for (i=start;i<m_numIndividuals;i++) {
			m_population[i] = new EDABitSet(m_numAttribs);
			BitSet temp = new BitSet(m_numAttribs);

			for (int a=0;a<m_numAttribs;a++) {
				if (a!=m_classIndex && m_random.nextDouble() < m_probVec[j][a]) {
					temp.set(a);
				}
			}

			temp.or(core);
			m_population[i].setSubset(temp);
		}

		return m_population;
	}


	/**
	 * generates a report on the current population, m_population
	 * @return a report as a String
	 */
	private String populationReport (int genNum, EDABitSet[] m_population) {
		int i;
		StringBuffer temp = new StringBuffer();

		if (genNum == 0) {
			temp.append("\nInitial population\n");
		}
		else {
			temp.append("\nGeneration: "+genNum+"\n");
		}
		temp.append("merit     \tsubset\n");

		for (i=0;i<m_numIndividuals;i++) {
			temp.append(Utils.doubleToString(Math.
					abs(m_population[i].getObjective()),
					8,5)
					+"\t");

			temp.append(printPopMember(m_population[i].getSubset())+" |"+m_population[i].getSubset().cardinality()+"|\n");
		}
		temp.append("Best subset found is "+printPopMember(m_gBest.getSubset())+" with merit: "+m_gBest.getObjective()+"\n");
		temp.append("Learning rate: "+m_learnRate+"\n");
		return temp.toString();
	}

	/**
	 * prints a population member as a series of attribute numbers
	 * @param temp the subset of a population member
	 * @return a population member as a String of attribute numbers
	 */
	private String printPopMember(BitSet temp) {
		StringBuffer text = new StringBuffer();

		for (int j=0;j<m_numAttribs;j++) {
			if (temp.get(j)) {
				text.append((j+1)+" ");
			}
		}
		return text.toString();
	}


	/**
	 * reset to default values for options
	 */
	private void resetOptions () {
		m_numberOfPopulations=1;
		m_numIndividuals = 200;
		m_learnRate = 0.1;
		m_negLearnRate = 0.075;
		m_maxGenerations = 30;
		m_reportFrequency = m_maxGenerations;
		m_seed = 1;
		
		m_lookupTableSize = 10001;
		
		gBest_merit = -999999999;
		gBest_cardinality=999999999;
		evals=0;
	}
}

