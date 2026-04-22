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


public class FuzzyGainRatio extends FuzzyMeasure implements TechnicalInformationHandler  {
	double denom;
	double decCardinality;
	double[] decisionLength,cards;
	public final static double LN2 = 0.69314718055994531;
	boolean m_isNumeric=false;

	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public FuzzyGainRatio() {
		super();
	}

	public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);

		m_isNumeric = m_trainInstances.attribute(m_classIndex).isNumeric();
		decisionLength = new double[m_numInstances];
		cards = new double[m_numInstances];

		for (int x = 0; x < m_numInstances; x++) {
			for (int y = 0; y < m_numInstances; y++) {
				double decS = fuzzySimilarity(m_classIndex, x, y);
				decisionLength[x] += decS;
			}

			decCardinality += 1 / decisionLength[x];
		}
	}

	@Override
	public double calculate(BitSet subset) {
		generatePartition(subset);
		double ret = 0;
		
		double classEntropy=0;
		double subsetEntropy=0;
		double joint=0;

		for (int x = 0; x < m_numInstances; x++) {
			double val=0,condval=0,decval = 0;

			for (int f = 0; f < m_numInstances; f++) {
				double condSim = current.getCell(x, f);
				double decSim = fuzzySimilarity(m_classIndex, x, f);

				condval += condSim;
				decval += decSim;

				val += m_TNorm.calculate(condSim, decSim);						
			}

			//mutual information
			//ret += log2((condval*decval)/(m_numInstances*val));
			//denom += log2(condval/m_numInstances)+log2(decval/m_numInstances);

			classEntropy += - (decval/m_numInstances)*log2(decval/m_numInstances);
			subsetEntropy += - (condval/m_numInstances)*log2(condval/m_numInstances);
			joint += - (val/m_numInstances)*log2(val/m_numInstances);
		}

		
		
//		ret/= -m_numInstances;
//		denom/=-m_numInstances;
		
		ret = (classEntropy+subsetEntropy - joint)/joint;
		
		System.err.println(subset+": "+ret+" "+joint);

		
		
		return (ret );

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
		return "Fuzzy gain ratio evaluation\n\n\n"
				+ getTechnicalInformation().toString();
	}

	public String toString() {
		return "Fuzzy gain ratio";
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}