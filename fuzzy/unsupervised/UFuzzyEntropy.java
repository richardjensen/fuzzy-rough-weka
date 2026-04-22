package weka.fuzzy.unsupervised;

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


public class UFuzzyEntropy extends UnsupervisedFuzzyMeasure implements TechnicalInformationHandler  {
	double denom;
	double decCardinality;
	double[] decisionLength,cards;
	public final static double LN2 = 0.69314718055994531;
	boolean m_isNumeric=false;

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public UFuzzyEntropy() {
		super();
	}


	public double calculate(BitSet subset, int decAttr) {
		generatePartition(subset);

		double ret = 0;
		double temp = 0;
		double cardinality = 0;
		decisionLength = new double[m_numInstances];
		cards = new double[m_numInstances];
		
		for (int x = 0; x < m_numInstances; x++) {
			for (int y = 0; y < m_numInstances; y++) {
				double decS = fuzzySimilarity(decAttr, x, y);
				decisionLength[x] += decS;
			}

			decCardinality += 1 / decisionLength[x];
		}
		
		
		for (int x = 0; x < m_numInstances; x++) {
			temp = 0;
			cards[x]=0;
			
			for (int d = 0; d < m_numInstances; d++) {
				double val = 0;
				double denom = 0;// temp=0;

				for (int y = 0; y < m_numInstances; y++) {
					double condSim = current.getCell(x, y);
					val += m_TNorm.calculate(condSim,fuzzySimilarity(decAttr, d, y));
					denom += condSim;
				}

				cards[x] = denom;
				temp += (-(val / denom) * log2(val / denom))/ decisionLength[d];
				
			}
			
			cardinality += cards[x];
			ret += cards[x] * temp;
		}
		ret /= cardinality;
		return 1 - (ret / log2(decCardinality));
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
		return "Unsupervised Fuzzy entropy evaluation.\n\n"
		+ getTechnicalInformation().toString();
	}

	public String toString() {
		return "Unsupervised Fuzzy entropy";
	}
}