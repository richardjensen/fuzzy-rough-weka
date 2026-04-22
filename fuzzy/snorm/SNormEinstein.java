package weka.fuzzy.snorm;

public class SNormEinstein extends SNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	
	public SNormEinstein(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return (a + b) / (1 + a * b);
	}

	public String getRevision() {
		return "Revision 1.0";
	}
	public String globalInfo() {
		return "Einstein S-norm: S(x,y) = (x + y) / (1 + x * y).";
	}
	
	public String toString() {
		return "Einstein";
	}
	
}
