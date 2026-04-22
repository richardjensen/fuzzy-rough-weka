package weka.fuzzy.measure;

import weka.core.Instances;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;

import java.io.Serializable;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.BitSet;

public abstract class nnFuzzyDiscernibilityMeasure extends nnFuzzyMeasure implements Serializable {
	public int fdm_length;
	static final long serialVersionUID = 1063626752258807303L;
	private HashMap<Integer,Clause> clauses;
	public boolean debug=false;
	public boolean m_ignoreDecision=false; //ignore the decision feature
	public ArrayList<Clause> clauseList=null;
	public BitSet core=null;


	public ArrayList<Clause> getClauses() {
		//return new ArrayList<Clause>(clauses.values());
		return clauseList;
	}	

	public void set(Similarity condSim, Similarity decSim, TNorm tnorm, TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs, int classIndex, Instances ins) {	
		super.set( condSim,  decSim,  tnorm,  compose,  impl,  snorm,  inst,  attrs,  classIndex,  ins);
		initialiseNeighbours(ins);
		calculateMatrix();
	}

	//Calculates the (fuzzy) discernibility matrix entries and stores them
	public void calculateMatrix() {
		System.err.print("Calculating matrix.");	
		clauses = new HashMap<Integer,Clause>();
		StringBuffer code;
		m_SNorm = (m_composition.getAssociatedSNorm());
		long start = System.currentTimeMillis();

		float totl = (1 * m_numInstances);
		float counter=0;
		core = new BitSet(m_numAttribs);

		for (int i = 0; i < m_numInstances; i++) {
			for (Integer j: neighbours[i]) {
				double[] values = new double[m_numAttribs];

				code = new StringBuffer("");
				double sum=0;
				double max=0;
				int setValues=0;
				int possibleCoreIndex=-1;

				if ((100*counter/totl) % 10 == 0) System.err.print(".");
				counter++;

				for (int a=0;a<m_numAttribs;a++) {
					values[a] = (1 - fuzzySimilarity(a,i,j));
					code.append(" "+values[a]);

					if (a!=m_classIndex) {
						sum+=values[a];
						max = m_SNorm.calculate(max, values[a]); 

						if (values[a]>0) {setValues++;possibleCoreIndex=a;} // == fuzzy discernibility>0

					}
				}

				if (m_ignoreDecision) values[m_classIndex]=1.0;

				if (values[m_classIndex]!=0) {
					max = m_Implicator.calculate(values[m_classIndex], max);

					if (setValues==1) core.set(possibleCoreIndex);

					Clause c = new Clause(values,code.toString().hashCode(),max,sum,setValues);

					int key = c.hashCode();
					if (!clauses.containsKey(key)) {
						if (simplify) determineAdd(key,c);
						else clauses.put(key, c);
					}
				}

				//ASSearch.pbar.setString("Calculating Discernibility Matrix...");
				//setProgressBar((int)(counter/totl));
			}
		}

		fdm_length = clauses.size(); 

		System.err.println("done");
		System.err.println("Core: "+core);
		System.err.println("Clauses in matrix: "+fdm_length+" out of a possible "+((m_numInstances*m_numInstances - m_numInstances)/2));
		System.err.println("Time taken for construction: "+(float)(System.currentTimeMillis() - start)/1000+" s");

		if (debug) printClauses();

		clauseList = new ArrayList<Clause>(clauses.values());
		java.util.Collections.sort(clauseList);


		//if full simplification isn't requested, perform a simple simplification procedure.
		if (!simplify) {
			int removed=0;
			int total=20;

			if (total>=clauses.size()) total=clauses.size()-1;

			System.err.print("Simplifying.");
			for (int p=0;p<total;p++) {
				int val=simplify2(p);
				if (val==-1) {
					total++;
					if (total >= clauses.size()) total=clauses.size()-1;
				}
				else {
					removed+=val;
					if (p%2==0) System.err.print(".");
				}
			}
			System.err.println("done");
			clauseList = new ArrayList<Clause>(clauses.values());

			System.err.println("Number of clauses: "+clauseList.size()+" (removed "+removed+" clauses)");
			clauses=null;
		}

		toRemove = new BitSet(m_numAttribs);
		coOccurrences();



		java.util.Collections.sort(clauseList);

		/*for (int a=0;a<3;a++) {
			Clause c = clauseList.get(a);
			System.err.println("First clause is "+c);
			System.err.println("Set values = "+c.setValues);
			System.err.println(compare(c,clauseList.get(a+1)));
		}*/
	}

	public BitSet toRemove=null;

	public BitSet coOccurrences() {
		Iterator<Clause> i;
		Clause o;
		int removeCount=0;


		for (int a=0; a<m_numAttribs;a++) {
			if (a!=m_classIndex) {

				for (int a2=0; a2<m_numAttribs;a2++) {
					if (a2!=m_classIndex && a!=a2 && !toRemove.get(a2)) {
						boolean remove=true;

						for (i = clauseList.iterator();i.hasNext();) {
							o = i.next();

							if (m_SNorm.calculate(o.getVariableValue(a),o.getVariableValue(a2))>o.getVariableValue(a2)) {
								remove=false;
								break;
							}
						}

						if (remove) {
							removeCount++;
							toRemove.set(a);
							break;
						}
					}
				}
			}
		}

		System.err.println("Can remove "+removeCount+" attribute(s) via (fuzzy) local strong compressibility: "+toRemove);
		return toRemove;
	}

	public int simplify2(int index) {
		Clause o=null;

		try{
			o = clauseList.get(index);
		}
		catch (Exception e) {

		}
		//Clause may have been removed by previous pruning
		if (o!=null) return prune(check(o));
		else return -1; //0
	}
	ArrayList<Integer> keys;


	/**
	 * For on-the-fly clause consideration
	 * @param key of the Clause
	 * @param c the Clause to consider adding to the set of Clause objects
	 */
	public void determineAdd(int key, Clause c) {
		boolean add=true;
		Iterator<Clause> it = clauses.values().iterator();
		keys = new ArrayList<Integer>();

		while (it.hasNext()) {
			Clause o2 = it.next();
			int dec = compare(c,o2);

			if (dec>=0) {
				add=false;
				break;
			} 
			else if (dec==-1) {
				o2.removed=true;
				keys.add(o2.hashCode);
			}

		}

		if (add) {
			Iterator<Integer> it2 = keys.iterator();

			while (it2.hasNext()) {
				clauses.remove(it2.next());
			}

			clauses.put(key, c);
		}
	}

	public boolean simplify=false;
	int startIndex=0;

	/**
	 * Loop through the clauses to see if one clause is subsumed by the other and therefore can be removed
	 * @param o
	 * @param o2
	 * @return
	 */
	private int compare(Clause o1, Clause o2) {
		double val1=0,cl1,cl2;

		boolean subset2=true;
		boolean subset1=true;
		startIndex=0;

		//if (o1.nonZeroIndex<o2.nonZeroIndex) startIndex = o1.nonZeroIndex;
		//else startIndex = o2.nonZeroIndex;

		for (int a=startIndex;a<m_numAttribs;a++) {
			cl1 = o1.getVariableValue(a);
			cl2 = o2.getVariableValue(a);
			val1 = m_TNorm.calculate(cl1,cl2);
			//val1 = Math.min(cl1,cl2);

			if (val1 != cl1) subset1=false;
			if (val1 != cl2) subset2=false;

			if (!subset1 && !subset2) break;
		}

		if (subset2) return 1; //clause o2 is a subset of clause o1, so o1 can be removed
		else if (subset1) return -1; //clause o1 is a subset of clause o2, so o2 can be removed
		else return -2;
	}

	private int check(Clause o) {
		int removedCount=0;
		Clause o2;

		Iterator<Clause> it = clauses.values().iterator();
		keys = new ArrayList<Integer>();

		//check to see if this clause is subsumed, if so then it is removed
		while (it.hasNext()) {
			o2 = it.next();

			if (o.hashCode!=o2.hashCode && !o2.removed) {
				int dec = compare(o,o2);

				//0 == identical
				//if (dec==0) {o.removed=true;removedCount++;break;}
				if (dec==1) {
					o.removed=true;
					removedCount++;
					keys.add(o.hashCode);
					break;
				}
				else if (dec==-1) {
					o2.removed=true;
					keys.add(o2.hashCode);
					removedCount++;
				}
			}
		}

		return removedCount;
	}


	/**
	 * Simplify the discernibility matrix
	 */
	/*private void simplify() {
		System.err.print("\nSimplifying...");
		int removedCount=0;


		Clause o;

		for (e1 = clauses.elements();e1.hasMoreElements();) {
			o = e1.nextElement();

			if (!o.removed) {			
				removedCount+=check(o);
			}
		}

		System.err.println("done");

		if (removedCount>0) {
			prune(removedCount);
		}
	}*/

	/**
	 * Loop through the clauses and prune away all those that have been flagged as 'removed'
	 * @param removedCount the number of clauses that will be removed
	 */
	private int prune(int removedCount) {
		Iterator<Integer> it = keys.iterator();

		while (it.hasNext()) {
			clauses.remove(it.next());
		}

		return removedCount;
	}

	private void printClauses() {
		Iterator<Clause> it1 = clauses.values().iterator();

		while (it1.hasNext()) {
			System.err.println(it1.next());
		}
	}
}
