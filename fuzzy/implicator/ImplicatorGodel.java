package weka.fuzzy.implicator;

public class ImplicatorGodel extends Implicator{
		/** for serialization. */
		private static final long serialVersionUID = 1065616253458807303L;
		
		public ImplicatorGodel(){
		  super();	
		}
		
		public double calculate(double a, double b) {
			if (a <= b) {
				return 1;
			} else {
				return b;
			}
		}

		public String getRevision() {
			return "Revision 1.0";
		}


		public String globalInfo() {
			return "The Godel implicator:\n"
			+ "I(x,y)=1 if x<=y, else I(x,y)=y";
			
		}
		public String toString() {
			return "Godel";
		}
	}
