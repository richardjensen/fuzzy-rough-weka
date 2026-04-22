package weka.fuzzy.snorm;

public class SNormLukasiewicz extends SNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	
	public SNormLukasiewicz(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return Math.min(1, a + b);
	}

	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "Lukasiewicz S-norm: S(x,y) = Math.min(1, x + y).";
		
	}
	
	public String toString() {
		return "Lukasiewicz";
	}
	
}
