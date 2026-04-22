package weka.fuzzy.measure;

import java.util.BitSet;

import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


public class WeakDelta extends FuzzyMeasure implements TechnicalInformationHandler {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;
	
	
	public WeakDelta() {
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

		
		for (int x = 0; x < m_numInstances; x++) {
			
			val = 0;
			int d = x;//optimization
			lower = 1;

			for (int y = 0; y < m_numInstances; y++) {
				double condSim = current.getCell(x, y);
				double decSim = fuzzySimilarity(m_classIndex, d, y);

				currLower = m_Implicator.calculate(condSim, decSim);

				lower = Math.min(currLower, lower);
				if (lower == 0) break;
			}

			val = lower;
			mems[x]=lower;
			
			ret = Math.min(ret, val);
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
		return "Weak delta evaluator, using the fuzzy rough lower approximations.\n\n"
		+ getTechnicalInformation().toString();
	}

	public String toString() {
		return "Weak delta";
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}
