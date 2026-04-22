package weka.fuzzy.measure;

import java.util.BitSet;
import java.util.PriorityQueue;
import java.util.Vector;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class WeakGammaOWAQ extends FuzzyMeasure  implements TechnicalInformationHandler {
	public double alpha=0.2;
	public double beta=1.0;


	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public WeakGammaOWAQ() {
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

		if (getModification()) {
			result.add("-L");
		}

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

	public boolean getModification() {
		return modification;
	}

	public void setModification(boolean m) {
		modification=m;
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

		setModification(Utils.getFlag('L', options));
	}

	public boolean modification=false;


	@Override
	public double calculate(BitSet subset) {
		generatePartition(subset);
		double ret = 0;

		for (int x = 0; x < m_numInstances; x++) {
			int d = x;
			PriorityQueue<Double> implications = new PriorityQueue<Double>();

			for (int y = 0; y < m_numInstances; y++) {
				double condSim = current.getCell(x, y);

				double decSim = fuzzySimilarity(m_classIndex, d, y);

				if (modification) {
					if (decSim!=1) 
						implications.add(m_Implicator.calculate(condSim, decSim));

				}
				else implications.add(m_Implicator.calculate(condSim, decSim));							
			}

			double sum = 0;
			int i = 0;
			int total = implications.size();

			for(Double implication: implications){							
				sum += implication*weight(total-i,alpha,beta,total);
				i++;
			}

			ret += sum;
		}

		return ret / n_objects_d;
	}

	public double weight(double i, double a, double b, double n){		
		return (q(i/n,a,b)-q((i-1)/n,a,b));

	}

	public double q(double x,double alpha,double beta ){
		if(x <=alpha)
			return 0;
		else if(alpha <=x && x<=((alpha+beta)/2))
			return ((2*(x-alpha)*(x-alpha))/((beta-alpha)*(beta-alpha)));
		else if (((alpha+beta)/2)<=x && x <=beta)
			return (1-((2*(x-beta)*(x-beta))/((beta-alpha)*(beta-alpha))));
		else if (beta <=x)
			return 1;
		else
			return 100000;

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
		return "Gamma evaluator, using the weak OWA operator for the lower approximation.\n\n"
		+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "WeakGammaOWA, alpha: "+ alpha + " beta: "+ beta;
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}