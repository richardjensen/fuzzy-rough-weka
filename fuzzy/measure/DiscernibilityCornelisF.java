package weka.fuzzy.measure;

import java.util.BitSet;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class DiscernibilityCornelisF extends FuzzyMeasure implements TechnicalInformationHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807403L;

	public DiscernibilityCornelisF() {
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
		double decSim;
		double oij;
		double ret;

		ret=1;

		for (int i = 0; i < m_numInstances - 1; i++) {
			for (int j = i + 1; j < m_numInstances; j++) {

				decSim = fuzzySimilarity(m_classIndex, i, j);			
				double saux;

				// optimization for qualitative decisions I(0,x)=1, I(x,1)=1
				if (decSim != 1) {
					double s = 0;
					// conditional attributes
					for (int a = reduct.nextSetBit(0); a >= 0; a = reduct.nextSetBit(a + 1)) {

						oij = m_Implicator.calculate(fuzzySimilarity(a, i, j),decSim);

						// numerator - S(O*_ij(a*)a*,...,)
						s = m_SNorm.calculate(oij, s);
						if (s==1) break;
					}
					saux = s;
				} else
					saux = 1;

				ret = m_TNorm.calculate(ret,saux);//Math.min(ret, saux);

			}
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
		return "Discernibility function (Cornelis) - f.\n\n"
		+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "Discernibility function (Cornelis) - f";
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}
