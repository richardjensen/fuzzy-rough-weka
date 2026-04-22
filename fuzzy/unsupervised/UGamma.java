package weka.fuzzy.unsupervised;


import java.util.BitSet;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class UGamma extends UnsupervisedFuzzyMeasure  implements TechnicalInformationHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public UGamma() {
		super();
	}
	

	
	
	@Override
	public double calculate(BitSet subset, int decAttr) {
		generatePartition(subset);
	
		double lower;
		double currLower = 0;
		double ret = 0;
		double val = 0;



		for (int x = 0; x < m_numInstances; x++) {
				val = 0;
				
				for (int d = 0; d < m_numInstances; d++) {
						lower = 1;

						// lower approximations of object x
						for (int y = 0; y < m_numInstances; y++) {
							double condSim = current.getCell(x, y);

							currLower = m_Implicator.calculate(condSim, fuzzySimilarity(decAttr, d, y));

							lower = Math.min(currLower, lower);
							if (lower == 0)	break;
						}
					
					val = Math.max(lower, val);

					if (val == 1) break;

				}

				

			ret += val;
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
		return "Unsupervised Gamma";
	}
}

