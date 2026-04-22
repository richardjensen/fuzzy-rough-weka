package weka.fuzzy.measure;

import java.util.BitSet;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Vector;

import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;


public class nnWeakGammaOWA extends nnFuzzyMeasure implements TechnicalInformationHandler  {
	public double alpha=0.2;
	public double beta=1.0;
	private Random rand = new Random();
	public int dummy=1; //to get around Weka Experimenter problems

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public nnWeakGammaOWA() {
		super();
	}

	public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);
		initialiseNeighbours(ins);
	}
	
	
	//Use the nearest neighbours (of a different class) to estimate the lower approximations
	public double calculate(BitSet subset) {
		if (knn<=0) return calculate(subset);
		double ret = 0;
		
		for (int x = 0; x < m_numInstances; x++) {
			int d = x;
			PriorityQueue<Double> implications = new PriorityQueue<Double>();

			for (Integer index: neighbours[x]) {
				int y = index;
				double condSim=1;
				for (int a = subset.nextSetBit(0); a >= 0; a = subset.nextSetBit(a + 1)) {
					if (a!=m_classIndex) {
						condSim = m_composition.calculate(fuzzySimilarity(a, x, y), condSim);

						if (condSim == 0) break;
					}
				}

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
		return "Nearest neighbour weak gamma evaluator, using the fuzzy rough lower approximations.\n\n"
				+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "Nearest neighbor OWA-based weak gamma";
	}


	/**
	 * Gets the current settings. Returns empty array.
	 *
	 * @return 		an array of strings suitable for passing to setOptions()
	 */
	public String[] getOptions(){
		Vector<String>	result;

		result = new Vector<String>();

		result.add("-T");
		result.add("" + rand.nextInt());

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
	

	//There is a problem in the Experimenter that means that to distinguish between
	//multiple runs of nnFDM, the dummy parameter should be changed
	/*public void setDummy(int d) {
		dummy=d;
	}
	
	public int getDummy() {
		return dummy;
	}*/

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
		
		/*tmpStr = Utils.getOption('T', options);
		if (tmpStr.length() != 0)
			setDummy(Integer.parseInt(tmpStr));
		else
			setDummy(0);*/
	}

	public boolean modification=false;

	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}

	
}
