package weka.fuzzy.measure;

import java.util.BitSet;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;


public class nnWeakGamma extends nnFuzzyMeasure implements TechnicalInformationHandler  {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public nnWeakGamma() {
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
		int d = -1;

		//for each instance x, loop over its neighbours and estimate the lower approximation
		for (int x=0; x<m_numInstances; x++) {
			double similarity = 1;
			double mem = 1;
			
			for (Integer index: neighbours[x]) {
				int y = index;
			
				for (int a = subset.nextSetBit(0); a >= 0; a = subset.nextSetBit(a + 1)) {
					if (a!=m_classIndex) {
						similarity = m_composition.calculate(fuzzySimilarity(a, x, y), similarity);

						if (similarity == 0) break;
					}
				}
				d = x;// weak version, so only consider x's decision

				mem = Math.min(mem, m_Implicator.calculate(similarity, fuzzySimilarity(m_classIndex, d, y)));				
			}
			
			ret+=mem;
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
		return "Nearest neighbour weak gamma evaluator, using the fuzzy rough lower approximations.\n\n"
				+  getTechnicalInformation().toString();
	}

	public String toString() {
		return "Nearest neighbor weak gamma";
	}

	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}

	
}
