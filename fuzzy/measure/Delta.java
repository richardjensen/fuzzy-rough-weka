package weka.fuzzy.measure;

import java.util.BitSet;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class Delta extends FuzzyMeasure implements TechnicalInformationHandler  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public Delta() {
		super();
	}
	
	/*public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);
	}*/
	
	@Override
	public double calculate(BitSet subset) {
		generatePartition(subset);
	
		double lower;
		double currLower = 0;
		double ret = 1;
		double val = 0;

		// for optimization
		indexes = new int[m_numInstances];
		vals = new double[m_numInstances];

		// optimization - see if other objects in the dataset are identical/have
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

						for (int y = 0; y < m_numInstances; y++) {
							double condSim = current.getCell(x, y);
							double decSim = fuzzySimilarity(m_classIndex, d, y);

							currLower = m_Implicator.calculate(condSim, decSim);

							lower = Math.min(currLower, lower);
							if (lower == 0) break;
						}
						decVals[d] = lower;
					}
					val = Math.max(decVals[d], val);
					if (val == 1) break;
				}
				vals[x] = val;
			}
			mems[x] = vals[x];
			ret = Math.min(ret, vals[x]);
			if (ret == 0 && !objectMemberships) break;
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

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "C. Cornelis, G. Hurtado Martin, R. Jensen and D. Slezak");
		result.setValue(Field.TITLE, "Feature Selection with Fuzzy Decision Reducts");
		result.setValue(Field.BOOKTITLE, "Third International Conference on Rough Sets and Knowledge Technology (RSKT'08)");
		result.setValue(Field.YEAR, "2008");
		result.setValue(Field.PAGES, "284-291");
		result.setValue(Field.PUBLISHER, "Springer");

		return result;
	}


	@Override
	public String globalInfo() {
		return "Delta evaluator, using the fuzzy rough lower approximations.\n\n"
		+ getTechnicalInformation().toString();
	}

	public String toString() {
		return "Delta";
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}
