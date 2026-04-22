package weka.fuzzy.tnorm;

import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.snorm.SNormAlgebraic;

public class TNormAlgebraic extends TNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	
	public TNormAlgebraic(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return a*b;
	}

	public String getRevision() {
		return "Revision 1.0";
	}
	public String globalInfo() {
		return "Algebraic T-norm: T(x,y) = xy.";
	}
	
	public String toString() {
		return "Algebraic";
	}
	
	public SNorm getAssociatedSNorm() {
		return new SNormAlgebraic();
	}
}
