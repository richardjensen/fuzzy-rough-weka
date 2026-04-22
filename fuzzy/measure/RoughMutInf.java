package weka.fuzzy.measure;

import java.util.BitSet;

import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;


public class RoughMutInf extends FuzzyMeasure implements TechnicalInformationHandler  {
	double denom;
	public final static double LN2 = 0.69314718055994531;
	boolean m_isNumeric=false;
	double norm =1;
	double correl[][];


	/**
	 * 
	 */
	private static final long serialVersionUID = 1063606253458807903L;

	public RoughMutInf() {
		super();
	}

	public void set(Similarity condSim, Similarity decSim, TNorm tnorm,
			TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs,
			int classIndex, Instances ins) {
		super.set(condSim, decSim, tnorm, compose, impl, snorm, inst, attrs, classIndex,
				ins);

		m_isNumeric = m_trainInstances.attribute(m_classIndex).isNumeric();
		correl = new double[m_numAttribs][m_numAttribs];

		for (int i=0;i<m_numAttribs;i++) {
			correl[i][i] = NOT_CALCULATED;

			for (int j=i+1;j<m_numAttribs;j++) {
				correl[i][j] = NOT_CALCULATED;
				correl[j][i] = NOT_CALCULATED;

			}	
		}
	}

	public static final int NOT_CALCULATED=-999;

	public int m_Method=0;
	
	public static final int METHOD_MRMR = 1;
	/** filter: No normalization/standardization */
	public static final int METHOD_CORREL = 0;
	/** The filter to apply to the training data */
	public static final Tag [] TAGS_METHOD = {
		new Tag(METHOD_MRMR, "Min redundancy, max relevancy"),
		new Tag(METHOD_CORREL, "Correlation-based approach"),
	};
	
	public SelectedTag getFilterType() {

		return new SelectedTag(m_Method, TAGS_METHOD);
	}

	/**
	 * Sets how the training data will be transformed. Should be one of
	 * FILTER_NORMALIZE, FILTER_STANDARDIZE, FILTER_NONE.
	 *
	 * @param newType the new filtering mode
	 */
	public void setFilterType(SelectedTag newType) {

		if (newType.getTags() == TAGS_METHOD) {
			m_Method = newType.getSelectedTag().getID();
		}
	}


	public double calculate(BitSet subset) {
		double cardinality = (double)subset.cardinality();

		double relevancy=0;
		double redundancy=0;
		
		for (int i = subset.nextSetBit(0); i >= 0; i = subset.nextSetBit(i+1)) {
			if (i!=m_classIndex){ 

				if (correl[i][m_classIndex]==NOT_CALCULATED) {
					correl[i][m_classIndex] = calculate(i,m_classIndex); //relevancy
				}

				relevancy+=correl[i][m_classIndex];

				for (int j = i+1;j<m_numAttribs;j++) {
					if (subset.get(j)&&j!=m_classIndex) {

						if (correl[i][j]==NOT_CALCULATED) {
							correl[i][j] = calculate(i,j);
							correl[j][i] = correl[i][j];
						}

						redundancy+= correl[i][j];
					}
				}
			}
		}
		
		//System.err.println(subset+": "+relevancy+" "+redundancy);

		
		//minRedundancy maxRelevancy
		if (m_Method==METHOD_MRMR) {
			relevancy/=cardinality;
			
			redundancy/=(cardinality*cardinality);

			
			return relevancy-redundancy;
		}
		else { //correlation-based approach
			double denom = cardinality + 2*redundancy;

			if (denom<0) denom*=-1;
	
			if (denom==0) return 0;
			else return (relevancy)/Math.sqrt(denom);
		}
	}

	//calculate the (rough) 'mutual information' between two attributes
	//the measure of granularity is actually the inverse granularity - lower values correspond to coarser partitions
	public double calculate(int a1, int a2) {
		double ret=0;
		double h1=0;
		double h2=0;
		double joint=0;
		double numInstances = (double)m_numInstances;

		for (int i=0;i<m_numInstances;i++) {
			double sum1=0;
			double sum2=0;
			double tnorm=0;

			for (int j=0;j<m_numInstances;j++) {
				double sim1 = fuzzySimilarity(a1, i, j);
				double sim2 = fuzzySimilarity(a2, i, j);

				sum1+=sim1;
				sum2+=sim2;
				//				sum tnorm of these
				tnorm += m_TNorm.calculate(sim1, sim2);
			}	

			h1 += numInstances - sum1;
			h2 += numInstances - sum2;
			joint += numInstances - tnorm;
			
		}
		
		h1 = (h1/(numInstances*(numInstances-1)));
		h2 = (h2/(numInstances*(numInstances-1)));
		joint = (joint/(numInstances*(numInstances-1)));
		
		//System.err.println(h1+ " "+h2+" "+ joint);
		
		ret = h1 + h2 - joint;

		return ret;
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
		return "Rough mutual information-based evaluation.\n\n"
		+ getTechnicalInformation().toString();
	}

	public String toString() {
		return "Rough mutual information";
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}