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


public class FuzzyEntropy extends FuzzyMeasure implements TechnicalInformationHandler  {
	double denom;
	double decCardinality;
	double[] decisionLength,cards;
	public final static double LN2 = 0.69314718055994531;
	boolean m_isNumeric=false;
	double norm=1; //for normalization


	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public FuzzyEntropy() {
		super();
	}

	public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);

		if (m_classIndex>=0) m_isNumeric = m_trainInstances.attribute(m_classIndex).isNumeric();
		else m_isNumeric=false;
		
		decisionLength = new double[m_numInstances];
		cards = new double[m_numInstances];
		
		for (int x = 0; x < m_numInstances; x++) {
			for (int y = 0; y < m_numInstances; y++) {
				double decS = fuzzySimilarity(m_classIndex, x, y);
				decisionLength[x] += decS;
			}

			//if (decisionLength[x]==0) decisionLength[x]=0.01;
			
			decCardinality += 1 / decisionLength[x];
			
		}

		
		norm=log2(decCardinality);
	}


	public double calculate(BitSet subset) {
		generatePartition(subset);

		double ret = 0;
		double temp = 0;
		double cardinality = 0;

		// for optimization
		indexes = new int[m_numInstances];
		vals = new double[m_numInstances];

		// optimization - see if other objects in the dataset are identical/have
		// identical similarity
		setIndexes(current);

		for (int x = 0; x < m_numInstances; x++) {
			temp = 0;
			// System.out.print(indexes[x]+" ");
			if (indexes[x] != 0) {
				cards[x] = cards[indexes[x] - 1];
				vals[x] = vals[indexes[x] - 1];
			} else {
				decVals = new double[m_numInstances];
				for (int d = 0; d < m_numInstances; d++) {
					if (decIndexes[d] != 0) {
						decVals[d] = decVals[decIndexes[d] - 1];
						temp += decVals[d];
					} else {
						double val = 0;
						double denom = 0;// temp=0;

						for (int f = 0; f < m_numInstances; f++) {
							double condSim = current.getCell(x, f);
							val += m_TNorm.calculate(condSim, fuzzySimilarity(m_classIndex, d, f));
							denom += condSim;
						}

						cards[x] = denom;

						decVals[d] = (-(val / denom) * log2(val / denom)) / decisionLength[d];
						temp += decVals[d];

						// optimization (for crisp decisions)
						if (cards[x] == 1 && !m_isNumeric) break;
					}
				}

				vals[x] = cards[x] * temp;

			}
			cardinality += cards[x];
			ret += vals[x];
			mems[x]=vals[x];
		}
		ret /= cardinality;
		return 1 - (ret / norm);
	}

	public double calculate(int a1, int a2) {
		double ret = 0;
		double temp = 0;
		double cardinality = 0;


		for (int x = 0; x < m_numInstances; x++) {
			temp = 0;

			decVals = new double[m_numInstances];
			for (int d = 0; d < m_numInstances; d++) {

				double val = 0;
				double denom = 0;// temp=0;

				for (int f = 0; f < m_numInstances; f++) {
					double condSim = fuzzySimilarity(a1, x, f);
					val += m_TNorm.calculate(condSim, fuzzySimilarity(a2, d, f));
					denom += condSim;
				}

				cards[x] = denom;

				decVals[d] = (-(val / denom) * log2(val / denom)) / decisionLength[d];
				temp += decVals[d];

				// optimization (for crisp decisions)
				if (cards[x] == 1 && !m_isNumeric) break;
			}
			
			vals[x] = cards[x] * temp;			
			cardinality += cards[x];
			ret += vals[x];
			mems[x]=vals[x];
		}
		ret /= cardinality;
		return 1 - (ret / norm);
	}

	public static final double log2(double x) {
		if (x > 0)
			return Math.log(x) / LN2;
		else
			return 0.0;
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
		result.setValue(Field.TITLE, "New Approaches to Fuzzy-rough Feature Selection");
		result.setValue(Field.JOURNAL, "IEEE Transactions on Fuzzy Systems");
		result.setValue(Field.SCHOOL, "Aberystwyth University");
		result.setValue(Field.ADDRESS, "");

		additional = result.add(Type.INPROCEEDINGS);
		additional.setValue(Field.AUTHOR, "N. Mac Parthalain, R. Jensen and Q. Shen");
		additional.setValue(Field.TITLE, "Finding Fuzzy-rough Reducts with Fuzzy Entropy");
		additional.setValue(Field.BOOKTITLE, "Proceedings of the 17th International Conference on Fuzzy Systems (FUZZ-IEEE'08)");
		additional.setValue(Field.YEAR, "2008");
		additional.setValue(Field.PAGES, "1282-1288");
		additional.setValue(Field.PUBLISHER, "IEEE");

		return result;
	}

	@Override
	public String globalInfo() {
		return "Fuzzy conditional entropy evaluation.\n\n"
		+ getTechnicalInformation().toString();
	}

	public String toString() {
		return "Fuzzy conditional entropy";
	}

	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}