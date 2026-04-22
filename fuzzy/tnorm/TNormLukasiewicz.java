package weka.fuzzy.tnorm;

import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormLukasiewicz;

public class TNormLukasiewicz extends TNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	
	public TNormLukasiewicz(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return Math.max(a+b-1,0);
	}

	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "Lukasiewicz T-norm: T(x,y) = max(x+y-1,0).";		
	}
	
	public String toString() {
		return "Lukasiewicz";
	}
	
	public SNorm getAssociatedSNorm() {
		return new SNormLukasiewicz();
	}
}
