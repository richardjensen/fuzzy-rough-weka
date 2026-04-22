/**
 * 
 */
package weka.attributeSelection;

import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.Debug.Clock;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.implicator.ImplicatorLukasiewicz;
import weka.fuzzy.measure.FuzzyMeasure;
import weka.fuzzy.measure.FuzzyMutInf;
import weka.fuzzy.measure.Measure;
import weka.fuzzy.similarity.*;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormKD;
import weka.fuzzy.tnorm.TNormLukasiewicz;
import weka.fuzzy.*;

import java.util.BitSet;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;
import java.util.Vector;
import java.util.ArrayList;

/**
 * @author ncm, rkj
 * 
 */
public class FuzMutFeatureGrouping4 extends ASSearch implements OptionHandler {

	/** for serialization */
	static final long serialVersionUID = -6312951970168325471L;

	/** does the data have a class */
	protected boolean m_hasClass;
	
	//if false, then use standard fuzzy mutual information
	public boolean useSymmetricUncertainty=true;

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	/** number of instances */
	protected int m_numInstances;


	/**
	 * A threshold by which to discard attributes---used by the
	 * AttributeSelection module
	 */
	protected double m_threshold;

	public Similarity m_Similarity = new Similarity3();
	public Similarity m_SimilarityEq = new SimilarityEq();
	public Similarity m_DecisionSimilarity = new SimilarityEq();
	public FuzzyMutInf fMI = new FuzzyMutInf();

	public TNorm m_TNorm = new TNormKD();
	public TNorm m_composition = new TNormLukasiewicz();
	public Implicator m_Implicator = new ImplicatorLukasiewicz();
	public SNorm m_SNorm = new SNormLukasiewicz();

	/** the merit of the best subset found */
	protected double m_bestMerit;

	/** a ranked list of attribute indexes */
	protected double[][] m_rankedAtts;
	protected int m_rankedSoFar;

	/** the subset evaluation measure */
	protected ASEvaluation m_ASEval = new FuzzyRoughSubsetEval();

	/** Threshold value, defaults to using the average as a cut-off point */
	protected double m_factor = -1;

	protected Instances m_Instances;

	/** holds an array of starting attributes */
	protected int[] m_starting;


	/**
	 * 
	 */
	public FuzMutFeatureGrouping4() {
		m_threshold = 1;
		m_starting = null;
		resetOptions();
	}

	/**
	 * Searches the attribute subset space by forward selection.
	 * 
	 * @param ASEval
	 *            the attribute evaluator to guide the search
	 * @param data
	 *            the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception
	 *             if the search can't be completed
	 */
	public int[] search(ASEvaluation ASEval, Instances data) throws Exception {
		m_numAttribs = data.numAttributes()-1;
		m_numInstances = data.numInstances();

		// if the data has no decision feature, m_classIndex is negative
		m_classIndex = data.classIndex();

		// Set params for FMI metric
		m_Similarity.setInstances(data);
		fMI.set(m_Similarity, m_DecisionSimilarity, m_TNorm, m_composition,
				m_Implicator, m_SNorm, m_numInstances, m_numAttribs,
				m_classIndex, data);

		if (data != null) { // this is a fresh run so reset
			resetOptions();
			m_Instances = data;
		}

		m_ASEval = ASEval;
		m_ASEval.buildEvaluator(data);

		//Used to evaluate subsets in the hill-climbing search
		SubsetEvaluator evaluator =  (SubsetEvaluator)ASEval;

		//the final reduct (built up in hill-climbing search later on)
		BitSet reduct = new BitSet(m_numAttribs);

		// Array for the features in the dataset
		Relation connectedGraph = new Relation(m_numAttribs);

		//average is used as a threshold later for determining groups
		double average = 0; 
		double denom = 0;		
		double[] decisionMutInf = new double[m_numAttribs];
	
		System.err.print("Calculating fMI...");
		for (int k = 0; k < m_numAttribs; k++) {
			for (int j = k; j < m_numAttribs; j++) {
				double mi;
				if (useSymmetricUncertainty) mi= fMI.symmetricUncertainty(k, j);
				else mi = fMI.calculate(k, j);
				
				connectedGraph.setCell(k, j, mi);
				connectedGraph.setCell(j, k, mi);	

				if (j!=m_classIndex && k!=m_classIndex) {
					average+=mi;
					denom++;
				}
			}
			//this is used for ordering preferences of features in groups later
			decisionMutInf[k] = fMI.calculate(k,m_classIndex);
		}
		System.err.println(".done");

		average/=denom;

		//default to using the average, but the user can supply a different value (between 0 and 1) as a threshold
		if (m_factor<0||m_factor>1) m_factor = average;

		System.out.println("Denom = "+denom);
		System.out.println("Av = "+average);
		System.out.println("Thresh = "+m_factor);
		
		//build groups
		//convert the groups into BitSets
		//this makes the search stage quicker
		HashMap<Integer, BitSet> hmap = new HashMap<Integer, BitSet>();

		for (int k = 0; k < m_numAttribs; k++) {
			BitSet group = new BitSet(m_numAttribs);
			group.set(k);
			
			for (int j = 0; j < m_numAttribs; j++) {
				double val = connectedGraph.getCell(j, k);

				if (val>=m_factor) {
					//add to the group
					group.set(j);
				}
			}			
			
			hmap.put(k, group);
		}

		//print out groups
		for (int a=0;a<m_numAttribs;a++) {
			System.err.println(a+": "+hmap.get(a));
		}

		boolean done = false;
		double current_best = -10000000;
		Random random = new Random();

		//work out the evaluation for the full dataset
		BitSet full = new BitSet(m_numAttribs);
		for (int a = 0; a < m_numAttribs; a++)
			if (a!=m_classIndex) full.set(a);

		double fullMeasure = evaluator.evaluateSubset(full);

		System.err.println("----------------------------");

		while (!done) {
			BitSet temp_group = (BitSet)reduct.clone();
			int best_attribute=-1;
			BitSet avoids = new BitSet(m_numAttribs);
			int iterations=0;

			//hill-climbing search using the groups
			for (int a=0; a<m_numAttribs;a++) {
				if (!temp_group.get(a) && !avoids.get(a)) {
					int attr=a;
					iterations++;

					//if this feature is part of a group, get the group and choose random feature
					if (hmap.get(a).cardinality()>1) {
						BitSet group = (BitSet)(hmap.get(a)).clone();
						group.andNot(temp_group);

						//instead of random here, could use decisionMutInf array to choose best feature in the group
						int index = random.nextInt(group.cardinality());
						
						for (int i = group.nextSetBit(0); i > 0; i = group.nextSetBit(i+1)) {	
							attr = i;
							index--;
							if (index<0) break;
						}			

						//avoid the other group members in this iteration
						avoids.or(group);
					}

					temp_group.set(attr);
					double temp_merit = evaluator.evaluateSubset(temp_group);
					temp_group.clear(attr);

					if (temp_merit>=current_best) {
						current_best = temp_merit;
						best_attribute=attr;
					}

				}
			}

			if (best_attribute==-1) {
				done = false;
				break;
			}
			else {
				System.err.println("Iterations: "+iterations+" (full = "+(m_numAttribs-reduct.cardinality())+")");
				reduct.set(best_attribute);
				System.err.println(reduct + " => "+current_best);
				if (current_best==fullMeasure) {done=false;break;}
			}
		}


		// Finally convert bitset to array so it can be returned
		int[] subSetArray = new int[reduct.cardinality()];
		int y = 0;
		for (int z = reduct.nextSetBit(0); z >= 0; z=reduct.nextSetBit(z + 1)) {
			subSetArray[y] = z;
			y++;
		}

		return subSetArray;

	}


	public Enumeration listOptions() {
		Vector<Option> newVector = new Vector<Option>(4);
		newVector.addElement(new Option("\tSimilarity relation.", "R", 1,
				"-R <val>"));
		newVector.addElement(new Option("\tConnectives" + ".", "C", 1,
				"-C <val>"));
		newVector.addElement(new Option("\tComposition" + ".", "F", 1,
				"-F <val>"));
		newVector.addElement(new Option("\tkNN" + ".", "K", 1, "-K <val>"));
		return newVector.elements();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		resetOptions();
		String optionString;
		
		optionString = Utils.getOption('I', options);
		if(optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if(moreOptions.length == 0) { 
				throw new Exception("Invalid Implicator specification string."); 
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setImplicator( (Implicator) Utils.forName( Implicator.class, className, moreOptions) );
		}
		else {
			setImplicator(new ImplicatorLukasiewicz());
		}
		
		optionString = Utils.getOption('T', options);
		if (optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if (moreOptions.length == 0) {
				throw new Exception("Invalid TNorm specification string.");
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setTNorm((TNorm) Utils.forName(TNorm.class, className,
					moreOptions));
		} else {
			setTNorm(new TNormLukasiewicz());
		}

		optionString = Utils.getOption('R', options);
		if (optionString.length() != 0) {
			String moreOptions[] = Utils.splitOptions(optionString);
			if (moreOptions.length == 0) {
				throw new Exception("Invalid Similarity specification string.");
			}
			String className = moreOptions[0];
			moreOptions[0] = "";

			setSimilarity((Similarity) Utils.forName(Similarity.class, className, moreOptions));
		} else {
			setSimilarity(new Similarity3());
		}

		optionString = Utils.getOption('K', options);

		if (optionString.length() != 0) {
			setFactor(Double.valueOf(optionString));
		}

		else {
			setFactor(-1);
		}
		
		setUseSymmetricUncertainty(Utils.getFlag('S', options));

	}

	/**
	 * Gets the current settings of FuzMutFeatureGrouping
	 * 
	 * @return an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();

		result.add("-I");
		result.add((m_Implicator.getClass().getName() + " " + Utils
				.joinOptions(m_Implicator.getOptions())).trim());

		result.add("-T");
		result.add((m_TNorm.getClass().getName() + " " + Utils
				.joinOptions(m_TNorm.getOptions())).trim());

		result.add("-R");
		result.add((m_Similarity.getClass().getName() + " " + Utils.joinOptions(m_Similarity.getOptions())).trim());

		result.add("-K");
		result.add(""+getFactor());

		if (getUseSymmetricUncertainty()) {
			result.add("-S");
		}
		
		return result.toArray(new String[result.size()]);
	}
	
	public void setUseSymmetricUncertainty(boolean b) {
		useSymmetricUncertainty=b;
	}

	public boolean getUseSymmetricUncertainty() {
		return useSymmetricUncertainty;
	}
	
	public void setFactor(double k) {
		m_factor = k;
	}

	public double getFactor() {
		return m_factor;
	}
	

	public void setImplicator(Implicator impl) {
		m_Implicator = impl;
	}

	public Implicator getImplicator() {
		return m_Implicator;
	}

	public TNorm getTNorm() {
		return m_TNorm;
	}

	public void setTNorm(TNorm tnorm) {
		m_TNorm = tnorm;
	}

	public void setSimilarity(Similarity sim) {
		m_Similarity = sim;

	}

	public Similarity getSimilarity() {
		return m_Similarity;
	}

	/**
	 * Resets options
	 */
	protected void resetOptions() {
		m_ASEval = null;
		m_Instances = null;
		m_rankedSoFar = -1;
		m_rankedAtts = null;
	}

	public static void main(String[] args) {

	}
	
	/**
	 * returns a description of the search
	 * @return a description of the search as a String
	 */
	public String toString() {
		StringBuffer desc = new StringBuffer();
		desc.append("Hill-climber search using feature grouping\n");
		if (useSymmetricUncertainty) desc.append("using (fuzzy) symmetric uncertainty ");
		else 	desc.append("using fuzzy mutual information ");
		desc.append("and threshold: "+m_threshold);

		return desc.toString();
	}

}
