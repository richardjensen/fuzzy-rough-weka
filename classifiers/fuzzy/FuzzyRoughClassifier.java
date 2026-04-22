package weka.classifiers.fuzzy;

import java.io.Serializable;
import java.util.Enumeration;

import weka.attributeSelection.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.similarity.Similarity1;
import weka.fuzzy.similarity.SimilarityEq;
import weka.fuzzy.tnorm.TNorm;
import weka.fuzzy.tnorm.TNormAlgebraic;

public abstract class FuzzyRoughClassifier extends AbstractClassifier implements Serializable, UpdateableClassifier {
	/** for serialization. */
	static final long serialVersionUID = -3080186098777067172L;
	/** The training instances used for classification. */
	protected Instances m_Train;

	/** The number of class values (or 1 if predicting numeric). */
	protected int m_NumClasses;

	/** The class attribute type. */
	protected int m_ClassType;

	/** The number of neighbours to use for classification (currently). */
	protected int m_kNN;
	
	/** This is used to specify an alternative attribute index to use as the 'class' **/
	protected int altClassIndex = -1; 

	protected boolean useFeatureWeighting=false;

	public double[] weightings; //weighting for each feature
	private static double MIN_WEIGHT=0.0001;

	public ASEvaluation ranker = new FuzzyRoughAttributeEval(); //ranker to use to generate the weightings

	public Similarity m_Similarity = new Similarity1();
	public Similarity m_SimilarityEq= new SimilarityEq();
	public TNorm m_composition = new TNormAlgebraic();

	public class Index implements Comparable<Index>{
		int object;
		double sim=0;

		public Index(int a, double g) {
			object = a;
			sim = g;
		}
		
		public int object() {
			return object;
		}

		//sort in descending order (best first)
		public int compareTo(Index o) {
			return Double.compare(o.sim,sim);
		}

		public String toString() {
			return object+":"+sim;
		}

	}

	/**
	 * Whether the value of k selected by cross validation has
	 * been invalidated by a change in the training instances.
	 */
	protected boolean m_kNNValid;

	public FuzzyRoughClassifier() {

	}

	/**
	 * Returns the tip text for this property.
	 * @return tip text for this property suitable for
	 * displaying in the explorer/experimenter gui
	 */
	public String KNNTipText() {
		return "The number of neighbours to use.";
	}

	/**
	 * Set the number of neighbours the learner is to use.
	 *
	 * @param k the number of neighbours.
	 */
	public void setKNN(int k) {
		m_kNN = k;
		m_kNNValid = false;
	}

	/**
	 * Gets the number of neighbours the learner will use.
	 *
	 * @return the number of neighbours.
	 */
	public int getKNN() {
		return m_kNN;
	}



	/**
	 * Get the number of training instances the classifier is currently using.
	 * 
	 * @return the number of training instances the classifier is currently using
	 */
	public int getNumTraining() {
		return m_Train.numInstances();
	}

	public abstract Enumeration<Option> listOptions();

	public Similarity getSimilarity() {
		return m_Similarity;
	}

	public void setSimilarity(Similarity s) {
		m_Similarity = s;
	}
	
	public void setRanker(ASEvaluation r) {
		ranker = r;
	}
	
	public ASEvaluation getRanker() {
		return ranker;
	}
	
	public void setUseFeatureWeighting(boolean b) {
		useFeatureWeighting = b;
	}

	public boolean getUseFeatureWeighting() {
		return useFeatureWeighting;
	}
	
	public int getAltClassIndex() {
		return altClassIndex;
	}

	// set an attribute to be the 'class' for missing value imputation
	public void setAltClassIndex(int i) {
		altClassIndex = i;
	}
	
	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if an error occurred during the prediction
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (m_Train.numInstances() == 0) {
			throw new Exception("No training instances!");
		}

		// if K is out of range, make it equal to the number of objects
		if (m_kNN <=0 || m_kNN > m_Train.numInstances()) {
			m_kNN = m_Train.numInstances();
		}

		Instances neighbours = kNearestNeighbours(instance, m_kNN);
		double[] distribution = makeDistribution( neighbours );

		return distribution;
	}

	/**
	 * Generates the classifier.
	 *
	 * @param instances set of instances serving as training data 
	 * @throws Exception if the classifier has not been generated successfully
	 */
	public void buildClassifier(Instances instances) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// remove instances with missing class
		instances = new Instances(instances);
		instances.deleteWithMissingClass();

		m_NumClasses = instances.numClasses();
		m_ClassType = instances.classAttribute().type();
		m_Train = new Instances(instances, 0, instances.numInstances());

		m_Similarity.setInstances(m_Train);
		m_composition = m_Similarity.getTNorm();

		// Invalidate any currently cross-validation selected k
		m_kNNValid = false;
		int numFeatures = m_Train.numAttributes();
		int m_classIndex = m_Train.classIndex();
		weightings = new double[numFeatures];

		if (useFeatureWeighting) {
			ranker.buildEvaluator(m_Train);

			if (ranker instanceof AttributeEvaluator) {
				AttributeEvaluator m_ranker = (AttributeEvaluator) ranker;
				
				//Next: find the maximum weighting and use to scale them??
				
				for (int a=0;a<numFeatures;a++) {
					if (a!=m_classIndex) {
						weightings[a] = m_ranker.evaluateAttribute(a);
						
						if (weightings[a]<=MIN_WEIGHT) weightings[a]=MIN_WEIGHT;
						else if (weightings[a]>1) weightings[a]=1;
						
						//weightings[a] = 1-weightings[a];
					}
				}

			}
			else {
				System.err.println("An AttributeEvaluator is required");
			}

		}
		else {
			for (int a=0;a<numFeatures;a++) weightings[a]=1;
		}
	}

	public abstract Instances kNearestNeighbours(Instance instance, int m_kNN);


	/**
	 * Adds the supplied instance to the training set.
	 *
	 * @param instance the instance to add
	 * @throws Exception if instance could not be incorporated
	 * successfully
	 */
	public void updateClassifier(Instance instance) throws Exception {

		if (m_Train.equalHeaders(instance.dataset()) == false) {
			throw new Exception("Incompatible instance types");
		}
		if (instance.classIsMissing()) {
			return;
		}
		System.err.println("update classifier called");
		m_Train.add(instance);
		m_kNNValid = false;
	}

	public final double fuzzySimilarity(int attr, double x, double y) {
		double ret = 0;			

		//no decision feature, so each object is distinct
		if (attr<0 && attr==m_Train.classIndex()) {
			ret=0;
		}
		else {
			//if it's the class attribute, use the class similarity measure
			//if it's a nominal attribute, then use crisp equivalence
			//otherwise use the general similarity measure
			if (Double.isNaN(x)||Double.isNaN(y)) ret=1;	
			else if (m_Train.attribute(attr).isNominal()) ret = m_SimilarityEq.similarity(attr, x, y);
			else ret = m_Similarity.similarity(attr, x, y);

		}

		return ret;
	}

	protected abstract double[] makeDistribution(Instances n) throws Exception;


}
