package weka.fuzzy.snorm;

public class SNormAlgebraic extends SNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	
	public SNormAlgebraic(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return (a + b) - (a * b);
	}

	public String getRevision() {
		return "Revision 1.0";
	}
	public String globalInfo() {
		return "Algebraic S-norm: S(x,y) = (x + y) - (x * y).";
	}
	
	public String toString() {
		return "Algebraic";
	}
	
}
