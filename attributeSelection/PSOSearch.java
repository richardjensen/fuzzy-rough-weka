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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Random;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * PSOSearch:<br/>
 * <br/>
 * Performs a search using PSO.<br/>
 * <br/>
 *
 * 
 *
 * @author Richard Jensen
 */
public class PSOSearch 
extends ASSearch 
implements OptionHandler, TechnicalInformationHandler {

	/** for serialization */
	static final long serialVersionUID = -1618264232838472679L;

	private int evals=0;
	
	/** does the data have a class */
	private boolean m_hasClass;

	/** holds the class index */
	private int m_classIndex;

	/** number of attributes in the data */
	private int m_numAttribs;

	/** the current population */
	private Particle [] m_population;

	/** the number of individual solutions */
	private int m_numParticles;

	/** the best population member found during the search */
	private Particle m_gBest;

	private double gBest_merit;
	private int gBest_cardinality;
	
	private double m_Weight=0.5;
	
	/** the number of entries to cache for lookup */
	protected int m_lookupTableSize;
	
	private int m_maxVelocity=10;

	/** the lookup table */
	private Hashtable<Integer, Double> m_lookupTable;

	/** random number generation */
	private Random m_random;

	/** seed for random number generation */
	private int m_seed;

	/** acceleration constant - local **/
	private double m_c1=1;
	
	/** acceleration constant - global **/
	private double m_c2=2;

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
	 * A particle for the PSO algorithm
	 */
	protected class Particle 
	implements Cloneable, Serializable {

		/** for serialization */
		static final long serialVersionUID = -2930607837482622824L;

		/** the position (== feature subset) */
		public BitSet subset;
		
		//memory of this particle's best subset so far
		public BitSet localBest;

		public double localBestMerit=0;
		
		//the velocity of this particle
		public int velocity;
		
		/** holds raw merit */
		private double m_objective = -Double.MAX_VALUE;
		
		/**
		 * Constructor
		 */
		public Particle (int attr) {
			subset = new BitSet(attr);
			velocity = 1;
		}

	
		
		/**
		 * makes a copy of this GABitSet
		 * @return a copy of the object
		 * @throws CloneNotSupportedException if something goes wrong
		 */
		public Object clone() throws CloneNotSupportedException {
			Particle temp = new Particle(subset.cardinality());
			temp.setObjective(this.getObjective());
			temp.setSubset((BitSet)(this.subset.clone()));
			temp.velocity = velocity;
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
			setNumParticles(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('G', options);
		if (optionString.length() != 0) {
			setMaxGenerations(Integer.parseInt(optionString));
			setReportFrequency(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption('A', options);
		if (optionString.length() != 0) {
			setC1((new Double(optionString)).doubleValue());
		}

		optionString = Utils.getOption('B', options);
		if (optionString.length() != 0) {
			setC2((new Double(optionString)).doubleValue());
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
			
	}


	public void setLookupTableSize(int n) {
		if (n<0) n=10;
		m_lookupTableSize=n;
	}
	
	public int getLookupTableSize() {
		return m_lookupTableSize;
	}

	/**
	 * Gets the current settings of PSOSearch.
	 *
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions () {
		String[] heuristicOptions = new String[0];


		String[] options = new String[17 + heuristicOptions.length+1];
		int current = 0;


		options[current++] = "-Z";
		options[current++] = "" + getNumParticles();
		options[current++] = "-G";
		options[current++] = "" + getMaxGenerations();
		options[current++] = "-A";
		options[current++] = "" + getC1();
		options[current++] = "-B";
		options[current++] = "" + getC2();
		options[current++] = "-R";
		options[current++] = "" + getReportFrequency();
		options[current++] = "-S";
		options[current++] = "" + getSeed();
		options[current++] = "-C";
		options[current++] = "" + getLookupTableSize();
		
		if (getPrune()) {
			options[current++] = "-P";
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
	 * get acceleration constant c1
	 * @return c1, the acceleration constant 
	 */
	public double getC1() {
		return m_c1;
	}

	/**
	 * set the acceleration constant c1
	 * @param a, the acceleration constant 
	 */
	public void setC1(double a) {
		m_c1 = a;
	}

	/**
	 * get acceleration constant c2
	 * @return c2, acceleration constant
	 */
	public double getC2() {
		return m_c2;
	}
	
/**
	 * set the acceleration constant c2
	 * @param b, acceleration constant
	 */
	public void setC2(double b) {
		m_c2 = b;
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
	public void setNumParticles(int p) {
		m_numParticles = p;
	}

	/**
	 * get the size of the population
	 * @return the population size
	 */
	public int getNumParticles() {
		return m_numParticles;
	}

	/**
	 * Returns a string describing this search method
	 * @return a description of the search suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return 
		"PSOSearch:\n\nPerforms a search using binary Particle Swarm Optimization.\n\n"
		+ "A number of particles are initialized at random locations (which correspond to feature subsets) and then swarm towards promising areas via the global"
		+ " best solution so far and each particle's local best. "
		+ "The smallest subset found overall with maximum quality is returned.\n\n"
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
	 * Constructor. Make a new PSOSearch object
	 */
	public PSOSearch() {
		resetOptions();

	}

	/**
	 * returns a description of the search
	 * @return a description of the search as a String
	 */
	public String toString() {
		StringBuffer desc = new StringBuffer();
		desc.append("\tPSO search.\n");


		desc.append("\tPopulation size: "+m_numParticles);
		desc.append("\n\tNumber of generations: "+m_maxGenerations);
		desc.append("\n\tC1: "
				+Utils.doubleToString(m_c1,6,3));
		desc.append("\n\tC2: "
				+Utils.doubleToString(m_c2,6,3));
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

		m_gBest = null;
		m_generationReports = new StringBuffer();

		if (!(ASEval instanceof SubsetEvaluator)) {
			throw  new Exception(ASEval.getClass().getName() 
					+ " is not a " 
					+ "Subset evaluator!");
		}
		

		if (ASEval instanceof UnsupervisedSubsetEvaluator) {
			m_hasClass = false;
		}
		else {
			m_hasClass = true;
			m_classIndex = data.classIndex();
		}

		SubsetEvaluator ASEvaluator = (SubsetEvaluator)ASEval;
		m_numAttribs = data.numAttributes();

		int extra = (int)((double)m_lookupTableSize*0.25);
		
		// initial random population
		m_lookupTable = new Hashtable<Integer, Double>(m_lookupTableSize+extra);
		keys = new ArrayList<Integer>(m_lookupTableSize);
		
		m_random = new Random(m_seed);
		m_population = new Particle [m_numParticles];		

		//set up the global best particle
		m_gBest = new Particle(m_numAttribs);
		gBest_merit = -999999999;
		gBest_cardinality=999999999;
		
		//limit the velocity of the particles
		m_maxVelocity = (int)(0.33333*m_numAttribs);
		BitSet core=new BitSet(m_numAttribs);
		
		if (ASEvaluator instanceof FuzzyRoughSubsetEval) {
			FuzzyRoughSubsetEval ev = (FuzzyRoughSubsetEval)ASEvaluator;
			if (ev.computeCore) {
				core = ev.computeCore();
				ev.core = core;
			}
		}
		
		
		
		
		//m_generationReports.append(populationReport(0));
		// set up random initial population
		initPopulation(ASEvaluator,core);
		double  max = (m_maxGenerations);
		double counter=0;
		
		for (int i=1;i<=m_maxGenerations;i++) {
			
			// for each particle...
			for (int p = 0; p < m_numParticles; p++) {
				evaluateParticle(ASEvaluator,m_population[p]);
						
			}
			
			for (int p = 0; p < m_numParticles; p++) {
				updateParticle(m_population[p]);			
			}
			
			counter++;
			
			if ((i == m_maxGenerations) || 
					((i % m_reportFrequency) == 0)) {
				m_generationReports.append(populationReport(i));
				
			}
			
			
		}
		
		System.err.println("Subset evaluations: "+evals+"\n");

		BitSet m_best_group = m_gBest.getSubset();
		double m_bestMerit = m_gBest.getObjective();
		
		//try removing features
		//(no point if we're already performing a backward search)
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
		
		m_lookupTable.clear();
		
		System.err.println("Size of reduct: "+m_best_group.cardinality()+"\n");
		return attributeList(m_best_group);
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

	private void updateParticle(Particle p) {
		int distance=0;
		int gdistance=0;
		
		//distance is the cardinality of XOR of subsets, i.e. the number of bits that differ between BitSets
		BitSet temp = (BitSet)p.subset.clone(); 
		temp.xor(p.localBest);
		distance = temp.cardinality();
		temp = (BitSet)p.subset.clone();
		temp.xor(m_gBest.subset);
		gdistance = temp.cardinality();
		
		/*System.err.println("1: "+distance+"  2:"+gdistance); distance=0;gdistance=0;
		for (int d=0;d<m_numAttribs;d++) {
			if (d!=m_classIndex) {
				if ((p.subset.get(d)||p.localBest.get(d))&&!(p.subset.get(d)&&p.localBest.get(d))) distance++;
				if ((p.subset.get(d)||m_gBest.get(d))&&!(p.subset.get(d)&&m_gBest.get(d))) gdistance++;
				
			}
		}
		System.err.println("3: "+distance+"  4:"+gdistance);*/

		int vel = (int)(m_Weight*p.velocity) + (int)(m_c1*m_random.nextDouble()*distance) + (int)(m_c2*m_random.nextDouble()*gdistance);
		
		if (vel>m_maxVelocity) vel=m_maxVelocity;
		
		/*if (vel>gdistance) {
			vel = vel-gdistance;
			p.subset.or(m_gBest.subset);
		}*/
		
		//if (vel<=gdistance) {
			for (int a=0;a<vel;a++) {
				int val = Math.abs(m_random.nextInt()) % (m_numAttribs);
				if (val!=m_classIndex) {
					if (m_gBest.get(val)) {
						if (p.subset.get(val)) {}//a--;
						else p.subset.set(val);
						
					}
					else p.subset.clear(val);
				}
				else a--;
			}
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
	private double evaluateParticle (SubsetEvaluator ASEvaluator, Particle particle)
	throws Exception {
		double merit;
		boolean reachedCapacity = (keys.size()>=m_lookupTableSize);
		int hashCode = particle.getSubset().hashCode();
				
		// if its not in the lookup table then evaluate and insert (if space)
		if (!m_lookupTable.containsKey(hashCode)) {
			merit = ASEvaluator.evaluateSubset(particle.getSubset());
			evals++;
			
			if (merit>1) merit=1;
			particle.setObjective(merit);
			
			//if the hashtable is full, replace a random element
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
			
			if ((merit > gBest_merit)||(merit==gBest_merit && particle.getSubset().cardinality() < gBest_cardinality)) {
				m_gBest = particle;
				gBest_merit = merit;
				gBest_cardinality = particle.getSubset().cardinality();
			}
		} else {
			merit = m_lookupTable.get(hashCode);
			particle.setObjective(merit);
		}
		
		//update the particle's local best
		if (particle.getObjective() > particle.localBestMerit) {
			particle.localBestMerit = particle.getObjective();
			particle.localBest = (BitSet)(particle.subset).clone();
		}
		
		return merit;
	}

	/**
	 * creates random population members for the initial population.
	 * 
	 * @throws Exception if the population can't be created
	 */
	private void initPopulation (SubsetEvaluator ASEvaluator, BitSet core) throws Exception {
		int i;
		int start = 0;

		for (i=start;i<m_numParticles;i++) {
			m_population[i] = new Particle(m_numAttribs);
			BitSet temp = new BitSet(m_numAttribs);
			randomReduct(temp);
			temp.or(core);
			m_population[i].setSubset(temp);
			m_population[i].localBest = (BitSet)temp.clone();
			randomVelocity(m_population[i]);
		}
	}
	
	//initialise the particle with a random velocity
	private final void randomVelocity(Particle p) {
		p.velocity = m_random.nextInt() % (m_maxVelocity);		
	}
	
	private final void randomReduct(BitSet s) {
		int val = m_classIndex;
		int numAttrs = randomAttribute();
			
		for (int a=0;a<numAttrs;a++) {
			val = Math.abs(m_random.nextInt()) % (m_numAttribs);
			if (val==m_classIndex) a--;
			else if (s.get(val)) a--;
			else {
				s.set(val);
			}
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
	private String populationReport(int genNum) {
		int i;
		StringBuffer temp = new StringBuffer();

		if (genNum == 0) {
			temp.append("\nInitial population\n");
		}
		else {
			temp.append("\nGeneration: "+genNum+"\n");
		}
		temp.append("merit     \tsubset\n");

		for (i=0;i<m_numParticles;i++) {
			temp.append(Utils.doubleToString(Math.
					abs(m_population[i].getObjective()),
					8,5)
					+"\t");

			temp.append(printPopMember(m_population[i].getSubset())+" |"+m_population[i].getSubset().cardinality()+"|\n");
		}
		temp.append("Best subset found is "+printPopMember(m_gBest.getSubset())+" with merit: "+m_gBest.getObjective()+"\n");
		
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
		evals=0;
		m_population = null;
		m_numParticles = 200;
		m_lookupTableSize = 10001;
		m_c1 = 1;
		m_c2 = 2;
		m_maxGenerations = 30;
		m_reportFrequency = m_maxGenerations;
		m_seed = 1;
		gBest_merit = -999999999;
		gBest_cardinality=999999999;
	}
}

