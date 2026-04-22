package weka.fuzzy.measure.old;

import java.util.BitSet;
import java.util.Vector;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.measure.FuzzyMeasure;
import weka.fuzzy.similarity.Relation;


public class WeakGammaVQRS2 extends FuzzyMeasure implements TechnicalInformationHandler {
	public double alpha=0.2;
	public double beta=1;
	/**
	 * A threshold
	 */
	protected double m_threshold=1.5;
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public WeakGammaVQRS2() {
		super();
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

		/*result.add("-T");
		result.add("" + getThreshold());*/
		
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
		
		/*tmpStr = Utils.getOption('T', options);
		if (tmpStr.length() != 0) {
			Double temp;
			temp = Double.valueOf(tmpStr);
			setThreshold(temp.doubleValue());
		}*/

	}
	
	@Override
	public double calculate(BitSet subset) {
		generatePartition(subset);
	
		double ret1 = 0;
		double ret2=0;
		double full=0;
		
		for (int x = 0; x < m_numInstances; x++) {
			double val1 = 0;
			double val2 = 0;

			val1 = lowerVQRS(current, x, x); // use object x's decision class only
			ret1 += val1;
			double lower=1;
			
			for (int y = 0; y < m_numInstances; y++) {

				lower = Math.min(m_Implicator.calculate(current.getCell(x, y), fuzzySimilarity(m_classIndex, x, y)), lower);

				if (lower == 0)	break;
			}
			val2 = lower;
			
			ret2+=val2;
			
			if (val1-val2>m_threshold) full+=Math.min(1,val1);
			else full+=val2;

		}
		return full / n_objects_d;
	}

	private final double lowerVQRS(Relation rel, int x, int d) {
		double val = 0;
		double denom = 0;

		// for each fuzzy equivalence class in the relation
		for (int f = 0; f < m_numInstances; f++) {
			double condSim = rel.getCell(x, f);
			val += m_TNorm.calculate(condSim, fuzzySimilarity(m_classIndex, d, f));

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
		TechnicalInformation result;

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
		// TODO Auto-generated method stub
		return "WeakGammaVQRS2 evaluator, using the vaguely quantified fuzzy rough lower approximations.\n\n"
		+ toString() + "\n\n" + getTechnicalInformation().toString();
	}

	public String toString() {
		return "WeakGammaVQRS2: alpha="+alpha+" beta="+beta;
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}
