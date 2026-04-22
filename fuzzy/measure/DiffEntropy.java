package weka.fuzzy.measure;

import java.util.BitSet;
import java.util.Vector;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class DiffEntropy extends FuzzyMeasure implements TechnicalInformationHandler  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public boolean m_useGranularity=false;

	public DiffEntropy() {
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

		if (getUseGranularity()) result.add("-G");

		return result.toArray(new String[result.size()]);
	}

	/**
	 * Parses a given list of options.
	 *
	 * @param options 	the list of options as an array of strings
	 * @throws Exception 	if an option is not supported
	 */
	public void setOptions(String[] options) throws Exception {
		setUseGranularity(Utils.getFlag('G', options));

	}

	public void setUseGranularity(boolean b) {
		m_useGranularity = b;
	}

	public boolean getUseGranularity() {
		return m_useGranularity;
	}

	double decGranularity=-1;

	public double calculate(BitSet subset) {
		generatePartition(subset);

		double lower;
		double currLower = 0;
		double ret = 0;
		int d = -1;

		double granularity=0;

		for (int x = 0; x < m_numInstances; x++) {
			d = x;// weak version, so only consider x's decision
			lower = 1;
			double sum1=0;

			// lower approximations of object x
			for (int y = 0; y < m_numInstances; y++) {
				currLower = m_Implicator.calculate(current.getCell(x, y), fuzzySimilarity(m_classIndex, d, y));
				sum1 += current.getCell(x,y);
				lower = Math.min(currLower, lower);

				if (lower == 0 && !getUseGranularity()) break;
			}

			granularity +=  sum1;

			//entropy approach G_6
			//granularity += (sum1/n_objects_d)*log2(sum1);

			mems[x] = lower;
			ret += lower;
		} 

		ret = ret / n_objects_d;

		//System.err.println(subset+" "+ret+" "+c_divisor);

		//if we're not considering the granularity or we're working out the consistency, just return gamma
		if (!getUseGranularity()) {
			return ret;
		}
		else if (subset.cardinality()==m_numAttribs-1) {
			c_divisor = ret;
			return ret;
		}
		else {
			//granularity measure G_4 in "Granularity of partitions" by Y Yao, L Zhao
			granularity/=(m_numInstances*m_numInstances);

			//granularity measure G_2
			//granularity/=(m_numInstances);

			if (decGranularity==-1) {
				for (int x = 0; x < m_numInstances; x++) {
					for (int y = 0; y < m_numInstances; y++) {
						decGranularity += fuzzySimilarity(m_classIndex, x, y);
					}
				}
				decGranularity/=(m_numInstances*m_numInstances);
			}	
			
			granularity = Math.abs(decGranularity-granularity);

			double combined;
			//combined = (ret/(1-granularity));
			combined = ret*granularity;
			
			//Uncomment this to return 1 when a reduct is found
			//if (ret/c_divisor == 1) return 1; //if it's a reduct, return 1
			//else return combined; //otherwise, factor in the granularity
			return combined;
		}

	}

	public static final double log2(double x) {
		if (x > 0)
			return Math.log(x) / 0.69314718055994531;
		else
			return 0.0;
	}

	//calculate dependency between two features
	public double calculate(int a1, int a2) {
		double lower;
		double currLower = 0;
		double ret = 0;
		int d = -1;

		for (int x = 0; x < m_numInstances; x++) {
			d = x;// weak version, so only consider x's decision
			lower = 1;

			// lower approximations of object x
			for (int y = 0; y < m_numInstances; y++) {
				currLower = m_Implicator.calculate(fuzzySimilarity(a1, d, y), fuzzySimilarity(a2, d, y));

				lower = Math.min(currLower, lower);

				if (lower == 0)
					break;
			}
			mems[x] = lower;
			ret += lower;

		} 
		return ret / n_objects_d;

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
		return "Differentiation entropy evaluator.\n\n"
				+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "Differentiation entropy";
	}

	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}
