package weka.fuzzy.implicator;

public class ImplicatorLukasiewicz extends Implicator {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807303L;
	
	public ImplicatorLukasiewicz(){
	  super();	
	}
	
	public double calculate(double a, double b){
		return Math.min(1, 1 - a + b);
	}

	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "The Lukasiewicz implicator:\n"
			 + "I(x,y) = min(1, 1 - x + y).";
		
	}
	public String toString() {
		return "Lukasiewicz";
	}
}
