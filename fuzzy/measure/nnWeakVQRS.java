package weka.fuzzy.measure;

import java.util.BitSet;
import java.util.Vector;

import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Relation;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;


public class nnWeakVQRS extends nnFuzzyMeasure implements TechnicalInformationHandler  {
	public double alpha=0.2;
	public double beta=1.0;

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public nnWeakVQRS() {
		super();
	}

	public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);
		initialiseNeighbours(ins);
	}
	
	public double calculate(BitSet subset) {
		double ret = 0;

		for (int x = 0; x < m_numInstances; x++) {
			double val = lowerVQRS(subset, x, x); // use object x's decision class only
			mems[x] = val;
			ret += val;

		}
		return Math.min(1, ret / n_objects_d);
	}

	//doesn't work yet!! - fuzzy similarity for the decision part is always 0 (these are neighbours of a different class), therefore this breaks down
	//Could consider all nearest neighbours?
	private final double lowerVQRS(BitSet subset, int x, int d) {
		double val = 0;
		double denom = 0;
		
		// for each neighbour
		for (Integer index: neighbours[x]) {
			int y = index;
			double condSim=1;
			for (int a = subset.nextSetBit(0); a >= 0; a = subset.nextSetBit(a + 1)) {
				if (a!=m_classIndex) {
					//System.err.println(fuzzySimilarity(a, x, y));
					condSim = m_composition.calculate(fuzzySimilarity(a, x, y), condSim);

					if (condSim == 0) break;
				}
			}
			//fuzzySimilarity(m_classIndex, d, y) is always zero
			val += m_TNorm.calculate(condSim, fuzzySimilarity(m_classIndex, d, y));
			denom += condSim;
		}
		return Q(val / denom, alpha, beta); // return quantified value
	}

	// fuzzy quantifier
	private final double Q(double x, double alpha, double beta) {
		double ret = 0;
		double denomVal = (beta - alpha) * (beta - alpha);

		if (x <= alpha)
			ret = 0;
		else if (x < ((alpha + beta) / 2)) {
			ret = (2 * (x - alpha) * (x - alpha)) / denomVal;
		} else if (x < beta) {
			ret = 1 - ((2 * (x - beta) * (x - beta)) / denomVal);
		} else if (beta <= x) {
			ret = 1;
		}

		return ret;

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
		result.setValue(Field.YEAR, "2009");
		result.setValue(Field.TITLE, "New Approaches to Fuzzy-rough Feature Selection. IEEE Transactions on Fuzzy Systems");
		result.setValue(Field.SCHOOL, "Aberystwyth University");

		additional = result.add(Type.INPROCEEDINGS);
		additional.setValue(Field.AUTHOR, "C. Cornelis, G. Hurtado Martin, R. Jensen and D. Slezak");
		additional.setValue(Field.TITLE, "Feature Selection with Fuzzy Decision Reducts");
		additional.setValue(Field.BOOKTITLE, "Third International Conference on Rough Sets and Knowledge Technology (RSKT'08)");
		additional.setValue(Field.YEAR, "2008");
		additional.setValue(Field.PAGES, "284-291");
		additional.setValue(Field.PUBLISHER, "Springer");

		return result;
	}

	@Override
	public String globalInfo() {
		return "Nearest neighbour weak gamma evaluator, using the fuzzy rough VQRS lower approximations.\n\n"
				+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "Nearest neighbor VQRS-based weak gamma";
	}


	/**
	 * Gets the current settings. Returns empty array.
	 *
	 * @return 		an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions(){
		Vector<String>	result;

		result = new Vector<String>();


		result.add("-A");
		result.add("" + getAlpha());

		result.add("-B");
		result.add("" + getBeta());

		return result.toArray(new String[result.size()]);
	}

	public void setAlpha(double a) {
		alpha = a;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setBeta(double b) {
		beta = b;
	}

	public double getBeta() {
		return beta;
	}


	/**
	 * Parses a given list of options.
	 *
	 * @param options 	the list of options as an array of strings
	 * @throws Exception 	if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		String	tmpStr;

		tmpStr = Utils.getOption('A', options);
		if (tmpStr.length() != 0)
			setAlpha(Double.parseDouble(tmpStr));
		else
			setAlpha(1.0);

		tmpStr = Utils.getOption('B', options);
		if (tmpStr.length() != 0)
			setBeta(Double.parseDouble(tmpStr));
		else
			setBeta(1.0);

	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}

	
}
