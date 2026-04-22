package weka.fuzzy.measure;

import java.util.BitSet;
import java.util.Vector;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class WeakGamma extends FuzzyMeasure implements TechnicalInformationHandler {

	// ... (Existing serialVersionUID, members, constructor, options handlers) ...
	private static final long serialVersionUID = 1063606253458807903L;
	public boolean m_useGranularity=false;
	public WeakGamma() { super(); }

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
		generatePartition(subset); // This now does all the heavy lifting (and uses the cache if available)

		double lower;
		double currLower = 0;
		double ret = 0;
		int d;

		double granularity = 0;

		for (int x = 0; x < m_numInstances; x++) {
			d = x; // weak version, so only consider x's decision
			lower = 1;
			double sum1 = 0;


			// For non-neighbours, current.getCell(x,y) will be 0, correctly resulting in an implicator value of 1.
			for (int y = 0; y < m_numInstances; y++) {
				currLower = m_Implicator.calculate(current.getCell(x, y), fuzzySimilarity(m_classIndex, d, y));
				sum1 += current.getCell(x, y);
				lower = Math.min(currLower, lower);

				if (lower == 0 && !getUseGranularity()) {
					break;
				}
			}

			granularity += sum1;
			mems[x] = lower;
			ret += lower;
		}

		ret = ret / n_objects_d;


		// ... (Rest of the granularity logic is unchanged) ...
		if (!getUseGranularity()) {
			return ret;
		}
		else if (subset.cardinality()==m_numAttribs-1) {
			c_divisor = ret;
			return ret;
		}
		else {
			granularity/=(m_numInstances*m_numInstances);
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
			combined = ret*granularity;
			return combined;
		}
	}


	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	public String globalInfo() {
		return "Weak gamma evaluator, using the fuzzy rough lower approximations.\n\n"
				+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "Weak gamma";
	}

	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}

	// ... (Rest of the file is unchanged: log2, calculate(a1, a2), getTechnicalInformation, etc.) ...
}