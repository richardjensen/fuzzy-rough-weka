package weka.attributeSelection;

import java.util.BitSet;

import weka.fuzzy.unsupervised.UnsupervisedFuzzyMeasure;

public interface USubsetEvaluator  {

	  /** for serialization */
	  static final long serialVersionUID = 627934376267488763L;
	  
	  /**
	   * evaluates a subset of attributes with respect to another attribute, a
	   *
	   * @param subset a bitset representing the attribute subset to be 
	   * evaluated 
	   * @return the "merit" of the subset
	   * @exception Exception if the subset could not be evaluated
	   */
	  public double evaluateSubset(BitSet subset, int a) throws Exception;
	  
	  public abstract UnsupervisedFuzzyMeasure getFuzzyMeasure();
}
