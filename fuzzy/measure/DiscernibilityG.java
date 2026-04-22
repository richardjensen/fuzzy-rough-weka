package weka.fuzzy.measure;

import java.util.BitSet;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class DiscernibilityG extends FuzzyMeasure implements TechnicalInformationHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606753458807903L;

	public DiscernibilityG() {
		super();
	}
	
	/*public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);
	}*/
	
	@Override
	public double calculate(BitSet reduct) {
		double difd;
		double disc, d;
		double ret;
		ret = 0;

		double valueRA = 0;

		for (int i = 0; i < m_numInstances - 1; i++) {
			for (int j = i + 1; j < m_numInstances; j++) {
				// decision attribute values
				difd = fuzzySimilarity(m_classIndex, i, j);
				valueRA = 1;

				// optimization
				if (difd != 1) {
					// conditional attributes in reduct
					for (int a = reduct.nextSetBit(0); a >= 0; a = reduct.nextSetBit(a + 1)) {
						disc = fuzzySimilarity(a, i, j);

						valueRA = m_composition.calculate(disc, valueRA);

						if (valueRA == 0) break;
					}
					d = m_Implicator.calculate(valueRA,difd);
				} else
					d = 1;

				ret += d;
			}
		}

		return ret / ((n_objects_d * n_objects_d - n_objects_d) / 2);
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
		// TODO Auto-generated method stub
		return "Discernibility function - g.\n\n"
		+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "Discernibility function - g";
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}
