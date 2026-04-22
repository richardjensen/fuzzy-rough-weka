package weka.classifiers.fuzzy;

import weka.classifiers.fuzzy.FuzzyRoughClassifier.Index;
import weka.core.Instance;

public abstract class IVFuzzyRoughClassifier extends FuzzyRoughClassifier {

/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public double param = 0.9;
	
	public class IVIndex implements Comparable<IVIndex>{
		int object;
		double[] sim;

		public IVIndex(int a, double[] g) {
			object = a;
			sim = g;
		}

		//sort in descending order (best first)
		public int compareTo(IVIndex o) {
			//need a better comparison
			
			//double lower = Double.compare(o.sim[0],sim[0]);
			//double upper = Double.compare(o.sim[1],sim[1]);
			
			return Double.compare((o.sim[0]+o.sim[1])/2,(sim[0]+sim[1])/2);
		}

		public String toString() {
			return object+": ["+sim[0]+","+sim[1]+"]";
		}

	}
	
	public final double[] IVfuzzySimilarity(int attr, Instance x, Instance y) {
		double[] ret = new double[2];			

		//no decision feature, so each object is distinct
		if (attr<0 && attr==m_Train.classIndex()) {
			ret[0]=0;
			ret[1]=0;
		}
		else {
			double mainVal = x.value(attr);
			double otherVal = y.value(attr);

			//if it's the class attribute, use the class similarity measure
			//if it's a nominal attribute, then use crisp equivalence
			//otherwise use the general similarity measure
			if (x.isMissing(attr) || y.isMissing(attr)) {
				ret[0]=0;
				ret[1]=1;
			}
			//else if (attr==m_Train.classIndex()) ret = m_DecisionSimilarity.similarity(attr, mainVal, otherVal,param);
			else if (m_Train.attribute(attr).isNominal()) ret = m_SimilarityEq.similarity(attr, mainVal, otherVal, param);
			else ret = m_Similarity.similarity(attr, mainVal, otherVal, param);

		}
		return ret;
	}
	
}
