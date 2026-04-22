package weka.fuzzy.snorm;

public class SNormDrastic extends SNorm {
	/** for serialization. */
	private static final long serialVersionUID = 1068606253458807903L;
	
	public SNormDrastic(){
	  super();	
	}
	
	public double calculate(double a, double b){
            if (a==0) {
                return b;
            }
            else if (b==0) {
                return a;
            }
                 else {
                    return 1;
                 }
	}

	public String getRevision() {
		return "Revision 1.0";
	}
	public String globalInfo() {
		return "Drastic S-norm: S(x,y) = (x, y, 1).";
	}
	
	public String toString() {
		return "Drastic";
	}
	
}
