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
 *    GeneticSearch.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package  weka.attributeSelection;

import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.fuzzy.similarity.Relation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Random;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * AntSearch:<br/>
 * <br/>
 * Performs a search using ACO.<br/>
 * <br/>
 *
 * 
 *
 * @author Richard Jensen
 */
public class AntSearch 
extends ASSearch 
implements OptionHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = -1618264232838472679L;

	/** holds the class index */
	private int m_classIndex;

	/** number of attributes in the data */
	private int m_numAttribs;

	/** the current population */
	private Ant [] m_population;

	/** the number of individual solutions */
	private int m_numAnts;

	/** the best population member found during the search */
	private Ant m_best;

	private double best_merit;
	private int best_cardinality;
	
	/** the number of entries to cache for lookup */
	private int m_lookupTableSize;

	/** the lookup table */
	private Hashtable<BitSet, Ant> m_lookupTable;

	/** random number generation */
	private Random m_random;

	/** seed for random number generation */
	private int m_seed;

	/** alpha value in the probabilistic transition rule (pheromone) **/
	private double m_Alpha=1;

	/** the subset evaluator to use for the heuristic measure**/
	private ASEvaluation m_HeuristicEvaluator = new FuzzyRoughSubsetEval();;
	
	/** beta value in the probabilistic transition rule (heuristic) **/
	private double m_Beta=2;

	double Q = 0.1; // used for pheromone update
	double q0 = 0.8; // used for a random deviation in an ant's path

	/** the maximum number of generations to evaluate */
	private int m_maxGenerations;

	/** how often reports are generated */
	private int m_reportFrequency;

	/** holds the generation reports */
	private StringBuffer m_generationReports;

	private Relation pheromone;
	private Relation heuristic;
	
	double subsetAlpha = 0;
	
	protected class Edge implements Serializable{
		int i; //from
		static final long serialVersionUID = -2930607857482622224L;
		
	    public Edge(int a) {
	    	i=a;
	    }
	    
	}
	
	// Inner class
	/**
	 * An Ant for the ACO algorithm
	 */
	protected class Ant 
	implements Cloneable, Serializable {

		/** for serialization */
		static final long serialVersionUID = -2930607837482622824L;

		/** the bitset */
		public BitSet subset;

		/** holds raw merit */
		private double m_objective = -Double.MAX_VALUE;

		/** the fitness */
		private double m_fitness;
		public ArrayList<Edge> edges; //the list of edges that make up this path
		public boolean finished;
		
		/**
		 * Constructor
		 */
		public Ant () {
			subset = new BitSet();
			finished=false;
			edges = new ArrayList<Edge>();
		}

		//add an edge to the ant's path
		public void addEdge(Edge e) {
			edges.add(e);
		}
		
		/**
		 * makes a copy of this GABitSet
		 * @return a copy of the object
		 * @throws CloneNotSupportedException if something goes wrong
		 */
		@SuppressWarnings("unchecked")
		public Object clone() throws CloneNotSupportedException {
			Ant temp = new Ant();

			temp.setObjective(this.getObjective());
			temp.setFitness(this.getFitness());
			temp.setSubset((BitSet)(this.subset.clone()));
			temp.edges = (ArrayList<Edge>)this.edges.clone();
			temp.finished = finished;
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
		 * get the subset
		 * @return the constructed subset of this population member
		 */
		public BitSet getSubset() {
			return subset;
		}

		/**
		 * set the subset
		 * @param c the subset to be set for this population member
		 */
		public void setSubset(BitSet c) {
			subset = c;
		}

		/**
		 * unset a bit in the subset
		 * @param bit the bit to be cleared
		 */
		public void clear(int bit) {
			subset.clear(bit);
		}

		/**
		 * set a bit in the subset
		 * @param bit the bit to be set
		 */
		public void set(int bit) {
			subset.set(bit);
		}

		/**
		 * get the value of a bit in the subset
		 * @param bit the bit to query
		 * @return the value of the bit
		 */
		public boolean get(int bit) {
			return subset.get(bit);
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
		
		if ((m_HeuristicEvaluator != null) && 
				(m_HeuristicEvaluator instanceof OptionHandler)) {
			      newVector.addElement(new Option("", "", 0, "\nOptions specific to " 
							      + "scheme " 
							      + m_HeuristicEvaluator.getClass().getName() 
							      + ":"));
			      @SuppressWarnings("unchecked")
				Enumeration<Option> enu = ((OptionHandler)m_HeuristicEvaluator).listOptions();

			      while (enu.hasMoreElements()) {
			        newVector.addElement(enu.nextElement());
			      }
			    }

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

		optionString = Utils.getOption('L', options);
	    if (optionString.length() == 0)
	      optionString = FuzzyRoughSubsetEval.class.getName();
	    setHeuristicEvaluator(ASEvaluation.forName(optionString,
					     Utils.partitionOptions(options)));
		
		optionString = Utils.getOption('Z', options);
		if (optionString.length() != 0) {
			setNumAnts(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('G', options);
		if (optionString.length() != 0) {
			setMaxGenerations(Integer.parseInt(optionString));
			setReportFrequency(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setAlpha((new Double(optionString)).doubleValue());
		}

		optionString = Utils.getOption('B', options);
		if (optionString.length() != 0) {
			setBeta((new Double(optionString)).doubleValue());
		}

		optionString = Utils.getOption('R', options);
		if (optionString.length() != 0) {
			setReportFrequency(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('S', options);
		if (optionString.length() != 0) {
			setSeed(Integer.parseInt(optionString));
		}
			
	}

	public void setHeuristicEvaluator(ASEvaluation forName) {
		m_HeuristicEvaluator =  forName;
	}
	
	public ASEvaluation getHeuristicEvaluator() {
		return m_HeuristicEvaluator;
	}

	/**
	 * Gets the current settings of ReliefFAttributeEval.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions () {
		String[] heuristicOptions = new String[0];

		if ((m_HeuristicEvaluator != null) && (m_HeuristicEvaluator instanceof OptionHandler)) {
			heuristicOptions = ((OptionHandler)m_HeuristicEvaluator).getOptions();
		}

		String[] options = new String[14 + heuristicOptions.length+1];
		int current = 0;

		if (getHeuristicEvaluator() != null) {
			options[current++] = "-L";
			options[current++] = "" + getHeuristicEvaluator().getClass().getName();
		}

		options[current++] = "-Z";
		options[current++] = "" + getNumAnts();
		options[current++] = "-G";
		options[current++] = "" + getMaxGenerations();
		options[current++] = "-A";
		options[current++] = "" + getAlpha();
		options[current++] = "-B";
		options[current++] = "" + getBeta();
		options[current++] = "-R";
		options[current++] = "" + getReportFrequency();
		options[current++] = "-S";
		options[current++] = "" + getSeed();

		if (heuristicOptions.length > 0) {
			//for (int i=0;i<heuristicOptions.length;i++) System.err.println(heuristicOptions[i]);

			options[current++] = "--";
			System.arraycopy(heuristicOptions, 0, options, current, heuristicOptions.length);
			current += heuristicOptions.length;
		}

		while (current < options.length) {
			options[current++] = "";
		}
		return  options;
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
	 * get how often repports are generated
	 * @return how often reports are generated
	 */
	public int getReportFrequency() {
		return m_reportFrequency;
	}

	
	/**
	 * get the probability of mutation
	 * @return the probability of mutation occurring
	 */
	public double getAlpha() {
		return m_Alpha;
	}

	/**
	 * set the parameter alpha for use in the probabilistic transition rule
	 * @param a the value of alpha
	 */
	public void setAlpha(double a) {
		m_Alpha = a;
	}

	/**
	 * get the probability of crossover
	 * @return the probability of crossover
	 */
	public double getBeta() {
		return m_Beta;
	}
	
/**
	 * set the parameter beta for use in the probabilistic transition rule
	 * @param b the value of beta
	 */
	public void setBeta(double b) {
		m_Beta = b;
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
	 * Returns the tip text for this property
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String populationSizeTipText() {
		return "Set the population size. This is the number of ants "
		+"(attribute sets) in the population.";
	}

	/**
	 * set the population size
	 * @param p the size of the population
	 */
	public void setNumAnts(int p) {
		m_numAnts = p;
	}

	/**
	 * get the size of the population
	 * @return the population size
	 */
	public int getNumAnts() {
		return m_numAnts;
	}

	/**
	 * Returns a string describing this search method
	 * @return a description of the search suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return 
		"AntSearch:\n\nPerforms a search using Ant Colony Optimization.\n\n"
		+ "For each generation, ants start off at a random feature and move probabilistically until there is no improvement in their constructed subset quality." 
		+ "The smallest subset found overall with maximum quality is returned.\n\n"
		+ "Pheromone is randomly distributed at first and updated locally.\n\n"
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
	    result.setValue(Field.YEAR, "2005");
	    result.setValue(Field.TITLE, "Fuzzy-Rough Data Reduction with Ant Colony Optimization");
	    result.setValue(Field.JOURNAL, "Fuzzy Sets and Systems");
	    result.setValue(Field.PAGES, "5-20");
	    result.setValue(Field.VOLUME, "149");
	    result.setValue(Field.NUMBER, "1");
	    
	    additional = result.add(Type.INPROCEEDINGS);
	    additional.setValue(Field.AUTHOR, "R. Jensen, Q. Shen");
	    additional.setValue(Field.TITLE, "Finding Rough Set Reducts with Ant Colony Optimization");
	    additional.setValue(Field.BOOKTITLE, "2003 UK Workshop on Computational Intelligence");
	    additional.setValue(Field.YEAR, "2003");
	    additional.setValue(Field.PAGES, "15-22");

		return result;
	}

	/**
	 * Constructor. Make a new AntSearch object
	 */
	public AntSearch() {
		resetOptions();

	}

	/**
	 * returns a description of the search
	 * @return a description of the search as a String
	 */
	public String toString() {
		StringBuffer AntString = new StringBuffer();
		AntString.append("\tAnt search.\n");


		AntString.append("\tPopulation size: "+m_numAnts);
		AntString.append("\n\tNumber of generations: "+m_maxGenerations);
		AntString.append("\n\tAlpha: "
				+Utils.doubleToString(m_Alpha,6,3));
		AntString.append("\n\tBeta: "
				+Utils.doubleToString(m_Beta,6,3));
		AntString.append("\n\tReport frequency: "+m_reportFrequency);
		AntString.append("\n\tRandom number seed: "+m_seed+"\n");
		AntString.append(m_generationReports.toString());
		return AntString.toString();
	}

	/**
	 * Searches the attribute subset space using an ACO algorithm.
	 *
	 * @param ASEval the attribute evaluator to guide the search
	 * @param data the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception if the search can't be completed
	 */
	public int[] search (ASEvaluation ASEval, Instances data)
	throws Exception {

		m_best = null;
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
		

		// initial random population
		m_lookupTable = new Hashtable<BitSet, Ant>(m_lookupTableSize);
		m_random = new Random(m_seed);
		m_population = new Ant [m_numAnts];		
		
		pheromone = new Relation(m_numAttribs);
		
		initPheromone();
		
		//use a different evaluator for the heuristic (must be a subset evaluator)
		m_HeuristicEvaluator.buildEvaluator(data);
		initHeuristic();
		
		m_best = new Ant();
		best_merit = -999999999;
		best_cardinality=999999999;
		
		BitSet core=new BitSet(m_numAttribs);
		
		if (ASEvaluator instanceof FuzzyRoughSubsetEval) {
			FuzzyRoughSubsetEval ev = (FuzzyRoughSubsetEval)ASEvaluator;
			if (ev.computeCore) {
				core = ev.computeCore();
				ev.core = core;
			}
		}
				
		for (int i=1;i<=m_maxGenerations;i++) {
			// set up random initial population
			initPopulation(ASEvaluator,core);
			
			// for each ant, if it hasn't finished then construct
			// a path
			for (int a = 0; a < m_numAnts; a++) {
				
				if (!m_population[a].finished)
					constructPath(ASEvaluator,m_population[a]);				
			}
			
			// update the pheromone (and add decay)
			updatePheromone();
			
			
			if ((i == m_maxGenerations) || 
					((i % m_reportFrequency) == 0)) {
				m_generationReports.append(populationReport(i));
				
			}
			
			
		}
		
		return attributeList(m_best.getSubset());
	}
	
	//just normalise at the moment (update is local)
	private void updatePheromone() {
		normalise(-1,pheromone);
	}
	
	/*
	* Generate pheromone matrix
	*/
	private void initPheromone() {
		double max = 0;
		
		for (int i = 0; i < m_numAttribs; i++) {
			if (i!=m_classIndex) {
				pheromone.setCell(i, i, 0.00001);

				for (int j = i + 1; j < m_numAttribs; j++) {
					if (j!=m_classIndex) {
						pheromone.setCell(i,j,m_random.nextDouble());

						if (pheromone.getCell(i,j) == 0 || pheromone.getCell(i,j) >= 1)
							pheromone.setCell(i,j,0.1);
						if (pheromone.getCell(i,j) > max)
							max = pheromone.getCell(i,j);
					}
				}
			}
		}
		normalise(max,pheromone);
	}
	
	private final void normalise(double m, Relation rel) {
		double max = m;

		if (max < 0) { // i.e. don't know the maximum
			for (int i = 0; i < m_numAttribs; i++) {
				for (int j = i; j < m_numAttribs; j++) {
					if (j!=m_classIndex && i!=m_classIndex) {
						if (rel.getCell(i,j) > max)
							max = rel.getCell(i,j);
					}
				}
			}
		}

		for (int i = 0; i < m_numAttribs; i++) {
			for (int j = i; j < m_numAttribs; j++) {
				if (j!=m_classIndex && i!=m_classIndex) rel.setCell(i,j,rel.getCell(i,j)/ max);
			}
		}
		// System.out.print(max+ " ");
	}
	
	private void initHeuristic() throws Exception{
		BitSet bs = new BitSet(m_numAttribs);
		heuristic = new Relation(m_numAttribs);
		double max=0;
		boolean attributeEval=false;
		SubsetEvaluator subsetEvaluator=null;
		AttributeEvaluator attributeEvaluator=null;
		
		if (m_HeuristicEvaluator instanceof AttributeEvaluator) {
			attributeEval=true;
			attributeEvaluator = (AttributeEvaluator) m_HeuristicEvaluator;
		}
		else subsetEvaluator = (SubsetEvaluator) m_HeuristicEvaluator;
		
		// work out the heuristics...
		for (int a = 0; a < m_numAttribs; a++) {
			if (a!=m_classIndex) {
				heuristic.setCell(a,a,-1);

				for (int j = a + 1; j < m_numAttribs; j++) {
					if (j!=m_classIndex) {
						// the matrix is symmetric 
						bs.set(a);
						bs.set(j);
						if (attributeEval) heuristic.setCell(j,a,attributeEvaluator.evaluateAttribute(a)+attributeEvaluator.evaluateAttribute(j));
						else heuristic.setCell(j,a,subsetEvaluator.evaluateSubset(bs));
						bs.clear(a);
						bs.clear(j);
						
						if (heuristic.getCell(j,a)>max) max = heuristic.getCell(j,a);
					}
				}	

			}
		}
		normalise(max,heuristic);
	}
	
	/**
	 * converts a BitSet into a list of attribute indexes 
	 * @param group the BitSet to convert
	 * @return an array of attribute indexes
	 **/
	private int[] attributeList(BitSet group) {
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

	/**
	 * evaluates a single ant. Population members are looked up in
	 * a hash table and if they are not found then they are evaluated using
	 * ASEvaluator.
	 * @param ASEvaluator the subset evaluator to use for evaluating population
	 * members
	 * @param antIndex the index of the ant to evaluate
	 * @throws Exception if something goes wrong during evaluation
	 */
	private double evaluateAnt(SubsetEvaluator ASEvaluator, Ant ant)
	throws Exception {
		double merit;

		// if its not in the lookup table then evaluate and insert
		if (m_lookupTable.containsKey(ant.getSubset()) == false) {
			merit = ASEvaluator.evaluateSubset(ant.getSubset());
			ant.setObjective(merit);
			m_lookupTable.put(ant.getSubset(),ant);
			
			if ((merit > best_merit)||(merit==best_merit && ant.getSubset().cardinality() <= best_cardinality)) {
				m_best = ant;//(Ant)ant.clone();
				best_merit = merit;
				best_cardinality = ant.getSubset().cardinality();
			}
		} else {
			Ant temp = m_lookupTable.get(ant.getSubset());
			merit = temp.getObjective();
			ant.setObjective(merit);
		}
		
		return merit;
	}
	
	private void constructPath(SubsetEvaluator ASEvaluator, Ant ant) throws Exception {
		double best_prob = -1;
		int best_a = -1;
		double prob, heur;
		double gam = -1;
		boolean going = true;
		double prevGam=0;
		
		while (going) {
			best_a = -1;
			best_prob = -1;
			int prev = (ant.edges.get(ant.edges.size() - 1)).i;
			
			// random deviation - promotes extra exploration
			// this happens according to the value q0 - a high value of
			// q0 will limit random detours (a value of q0=1 will prevent
			// it altogether).
			if (Math.abs(m_random.nextDouble()) > q0) {
				while (true) {
					int poss = Math.abs(m_random.nextInt()) % (m_numAttribs);
					best_a = poss;
					if (best_a!=m_classIndex && !ant.getSubset().get(best_a))
						break;
					
				}
				best_prob = 1;
			} else {
				// for each attribute, work out the value of the
				// probabilistic transition rule, choose the attribute
				// that produces its highest value
				for (int a = 0; a < m_numAttribs; a++) {
					if (a!=m_classIndex && !ant.getSubset().get(a)) { // add an attribute
						BitSet temp = (BitSet) ant.getSubset().clone();
						temp.set(a);

						// get the heuristic value
						heur = heuristic.getCell(a, prev);
						if (heur == 0)
							heur = 0.000000001;

						// work out the probability using the heuristic
						// and pheromone levels
						prob = Math.pow(pheromone.getCell(a,prev), m_Alpha)
								* Math.pow(heur, m_Beta);

						// if this probability is better than the best so far
						// then consider this attribute to be the best
						if (prob > best_prob) {
							best_prob = prob;
							best_a = a;
						}

					}
				}
			}
			// if no best attribute has been found, then break
			if (best_a == -1) {
				break;
			}

			// set this attribute (it's the best one)
			ant.subset.set(best_a);
			ant.addEdge(new Edge(best_a)); // remember path

			// evaluate the reduct
			gam = evaluateAnt(ASEvaluator, ant);

			if (gam >= subsetAlpha) {
				subsetAlpha=gam;
				ant.finished = true;
				going = false;
			} 
			else if (prevGam > gam) {
				going=false;
				ant.finished=true;
			}
			prevGam = gam;
			
			//update pheromone locally
			double val = (1 - gam) * 0.5 + gam * pheromone.getCell(best_a,prev);
			pheromone.setCell(best_a,prev, val); 
			
		}
	}

	/**
	 * creates random population members for the initial population.
	 * 
	 * @throws Exception if the population can't be created
	 */
	private void initPopulation (SubsetEvaluator ASEvaluator,BitSet core) throws Exception {
		int i;
		int start = 0;

		for (i=start;i<m_numAnts;i++) {
			m_population[i] = new Ant();
			BitSet temp = new BitSet(m_numAttribs);
			int att = randomAttribute();
			temp.set(att);
			m_population[i].addEdge(new Edge(att));
			temp.or(core);
			m_population[i].setSubset(temp);
			evaluateAnt(ASEvaluator,m_population[i]);
		}
	}
	
	// returns a random conditional attribute number
	private final int randomAttribute() {
		int val = m_classIndex;
		
		while (val==m_classIndex) {
			val = Math.abs(m_random.nextInt()) % (m_numAttribs); 	
		}
		
		return val;
	}

	/**
	 * generates a report on the current population
	 * @return a report as a String
	 */
	private String populationReport (int genNum) {
		int i;
		StringBuffer temp = new StringBuffer();

		if (genNum == 0) {
			temp.append("\nInitial population\n");
		}
		else {
			temp.append("\nGeneration: "+genNum+"\n");
		}
		temp.append("merit     \tsubset\n");

		for (i=0;i<m_numAnts;i++) {
			temp.append(Utils.doubleToString(Math.
					abs(m_population[i].getObjective()),
					8,5)
					+"\t");

			temp.append(printPopMember(m_population[i].getSubset())+"\n");
		}
		temp.append("Best subset found is "+printPopMember(m_best.getSubset())+" with merit: "+m_best.getObjective()+"\n");
		
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
		m_population = null;
		m_HeuristicEvaluator=new FuzzyRoughSubsetEval();
		m_numAnts = 10;
		m_lookupTableSize = 7001;
		m_Alpha = 1;
		m_Beta = 2;
		m_maxGenerations = 10;
		m_reportFrequency = m_maxGenerations;
		m_seed = 1;
		best_merit = -999999999;
		best_cardinality=999999999;
	}
}

