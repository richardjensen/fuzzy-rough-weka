package weka.fuzzy.similarity;

import weka.core.*;
import weka.core.TechnicalInformation.Type;


public class SimilarityEq 
extends Similarity
implements Cloneable, TechnicalInformationHandler {

	/** for serialization. */
	private static final long serialVersionUID = 1066606253458807903L;



	/**
	 * Constructs Similarity based Distance object, Instances must be still set.
	 */
	public SimilarityEq() {
		super();
	}

	/**
	 * Constructs an Similarity based Distance object and automatically initializes the
	 * ranges.
	 * 
	 * @param data 	the instances the distance function should work on
	 */
	public SimilarityEq(Instances data) {
		super(data);
	}

	/**
	 * Returns a string describing this object.
	 * 
	 * @return 		a description of the evaluator suitable for
	 * 			displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {
		return 
		"The similarity measure is: equivalence";
	}

	public String toString() {
		return "Equivalence";
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing 
	 * detailed information about the technical background of this class,
	 * e.g., paper reference or book this class is based on.
	 * 
	 * @return 		the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation 	result;

		result = new TechnicalInformation(Type.MISC);
		//to be filled in
		return result;
	}


	/**
	 * similarity function calculates similarity between 2 attribute values
	 * 
	 * @param index   the index of the attribute for which the similarity is calculated
	 * @param first 	the value of the attribute of the first instance
	 * @param second 	the value of the attribute of the second instance
	 * 
	 * @return 	a measurement of similarity
	 * 
	 * 
	 */
	public double similarity(int index, double first, double second){
		if (first==second) return 1;
		else return 0;
	}
	
	//Interval-valued version
	  public double[] similarity(int index, double first, double second, double param){
		  double[] ret = new double[2];
		  
		  if (first==second) {
			  ret[0] = ret[1] = 1;
		  }
		  else {
			  ret[0] = ret[1] = 0;
		  }
		 
		  return ret;
	  }

	@Override
	public void clean() {
		// TODO Auto-generated method stub
		
	}


}


