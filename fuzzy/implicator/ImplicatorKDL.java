package weka.fuzzy.implicator;

public class ImplicatorKDL extends Implicator {
	/** for serialization. */
	private static final long serialVersionUID = 1063606253458807303L;

	public ImplicatorKDL() {
		super();	
	}

	public double calculate(double a, double b){
		return 1 - a + a*b;
	}

	public String getRevision() {
		return "Revision 1.0";
	}


	public String globalInfo() {
		return "The K-D-Lukasiewicz implicator:\n"
		+ "I(x,y) =  1 - x + xy";

	}
	public String toString() {
		return "K-D-Lukasiewicz (Reichenbach)";
	}
}


