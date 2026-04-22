package weka.fuzzy.unsupervised;

import java.util.BitSet;
import java.util.Vector;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.similarity.Relation;


public class UVQRS extends UnsupervisedFuzzyMeasure  implements TechnicalInformationHandler {
	public double alpha=0.2;
	public double beta=1;

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public UVQRS() {
		super();
	}

	/*public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);
	}*/

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

	@Override
	public double calculate(BitSet subset, int decAttr) {
		generatePartition(subset);

		double ret = 0;

		for (int x = 0; x < m_numInstances; x++) {
			double val = 0;

			// positive region
			for (int d = 0; d < m_numInstances; d++) {								
				val = Math.max(lowerVQRS(current, x, d, decAttr), val);
				if (val == 1) break;
			}
			
			ret += val;
		}

		return Math.min(1, ret / n_objects_d);
	}

	private final double lowerVQRS(Relation rel, int x, int d, int decAttr) {
		double val = 0;
		double denom = 0;

		// for each fuzzy equivalence class in the relation
		for (int y = 0; y < m_numInstances; y++) {
			double condSim = rel.getCell(x, y);
			val += m_TNorm.calculate(condSim, fuzzySimilarity(decAttr, d, y));

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

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "C. Cornelis and R. Jensen");
		result.setValue(Field.TITLE, "A Noise-tolerant Approach to Fuzzy-Rough Feature Selection");
		result.setValue(Field.BOOKTITLE, "17th International Conference on Fuzzy Systems (FUZZ-IEEE'08)");
		result.setValue(Field.YEAR, "2008");
		result.setValue(Field.PAGES, "1598-1605");
		result.setValue(Field.PUBLISHER, "IEEE");

		return result;
	}

	@Override
	public String globalInfo() {
		return "Unsupervised VQRS evaluator, using the vaguely quantified fuzzy rough lower approximations.\n\n"
		+ toString() + "\n\n" + getTechnicalInformation().toString();
	}

	public String toString() {
		return "Unsupervised VQRS: alpha="+alpha+" beta="+beta;
	}
}
