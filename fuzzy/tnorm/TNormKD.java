package weka.fuzzy.tnorm;

import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormKD;

public class TNormKD extends TNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	
	public TNormKD(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return Math.min(a,b);
	}

	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "Kleene-Dienes T-norm: T(x,y) = min(x,y).";
	}
	public String toString() {
		return "Kleene-Dienes";
	}
	
	public SNorm getAssociatedSNorm() {
		return new SNormKD();
	}
}
