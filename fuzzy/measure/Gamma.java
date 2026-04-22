package weka.fuzzy.measure;

import java.util.BitSet;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class Gamma extends FuzzyMeasure  implements TechnicalInformationHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public Gamma() {
		super();
	}
	
	
	@Override
	public double calculate(BitSet subset) {
		generatePartition(subset);
	
		double lower;
		double currLower = 0;
		double ret = 0;
		double val = 0;

		// for optimization
		indexes = new int[m_numInstances];
		vals = new double[m_numInstances];

		// optimisation - see if other objects in the dataset are identical/have
		// identical similarity
		setIndexes(current);

		for (int x = 0; x < m_numInstances; x++) {
			if (indexes[x] != 0) {
				vals[x] = vals[indexes[x] - 1];
			} else {
				val = 0;
				decVals = new double[m_numInstances];
				for (int d = 0; d < m_numInstances; d++) {
					if (decIndexes[d] != 0) {
						decVals[d] = decVals[decIndexes[d] - 1];
					} else {
						lower = 1;

						// lower approximations of object x
						for (int y = 0; y < m_numInstances; y++) {
							double condSim = current.getCell(x, y);

							currLower = m_Implicator.calculate(condSim, fuzzySimilarity(m_classIndex, d, y));

							lower = Math.min(currLower, lower);
							if (lower == 0)
								break;
						}


						decVals[d] = lower;
					}
					val = Math.max(decVals[d], val);

					if (val == 1) break;

				}

				vals[x] = val;

			}

			ret += vals[x];
			mems[x]=vals[x];
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
		return "Gamma evaluator, using the fuzzy rough lower approximations.\n\n"
		+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "Gamma";
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}
