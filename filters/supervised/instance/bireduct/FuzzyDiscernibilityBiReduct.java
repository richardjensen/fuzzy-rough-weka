package weka.filters.supervised.instance.bireduct;

import weka.core.Instances;
import weka.fuzzy.implicator.Implicator;
import weka.fuzzy.measure.*;
import weka.fuzzy.similarity.Similarity;
import weka.fuzzy.snorm.SNorm;
import weka.fuzzy.tnorm.TNorm;

import java.io.Serializable;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.BitSet;

public abstract class FuzzyDiscernibilityBiReduct extends FuzzyMeasure implements Serializable {
	public int fdm_length;
	static final long serialVersionUID = 1063626752258807303L;
	private HashMap<Integer,Clause> clauses;
	public boolean debug=false;
	public boolean m_ignoreDecision=false; //ignore the decision feature
	public ArrayList<Clause> clauseList=null;
	public BitSet core=null;
	//public Instances m_instances; 
	
	public void setInstances(Instances instances) {
		//this.m_instances = instances;
	}
	
	
	public ArrayList<Clause> getClauses() {
		//return new ArrayList<Clause>(clauses.values());
		return clauseList;
	}

	public void set(Similarity condSim, Similarity decSim, TNorm tnorm, TNorm compose, Implicator impl, SNorm snorm, int inst, int attrs, int classIndex, Instances ins) {	
		super.set( condSim,  decSim,  tnorm,  compose,  impl,  snorm,  inst,  attrs,  classIndex,  ins);
		calculateMatrix();
	}
	
	
	//Calculates the (fuzzy) discernibility matrix entries and stores them
	public ArrayList<Clause> calculateMatrix() {
		System.err.print("Calculating matrix.");	
		clauses = new HashMap<Integer,Clause>();
		StringBuffer code;
		m_SNorm = (m_composition.getAssociatedSNorm());
		long start = System.currentTimeMillis();
		float totl = (m_numInstances*m_numInstances - m_numInstances)/2;
		float counter=0;
		core = new BitSet(m_numAttribs);
		
		for (int i = 0; i < m_numInstances; i++) {
			for (int j = i + 1; j < m_numInstances; j++) {

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
				
				code.append(" " + i);
				code.append(" " + j);
				
				if (m_ignoreDecision) values[m_classIndex]=1.0;

				if (values[m_classIndex]!=0) {
					max = m_Implicator.calculate(values[m_classIndex], max);
					if (setValues==1) core.set(possibleCoreIndex);
					
					Clause c = new Clause(values,code.toString().hashCode(),max,sum,setValues);
					c.setObjects(i, j);
					
					c.setClasses((int)m_trainInstances.get(i).classValue(), (int)m_trainInstances.get(j).classValue());
					int key = c.hashCode();
					if (!clauses.containsKey(key)) {
						//if (simplify) determineAdd(key,c);
						//else 
						clauses.put(key, c);
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
		//java.util.Collections.sort(clauseList);


		//if full simplification isn't requested, perform a simple simplification procedure.
		/*if (!simplify) {
			int removed=0;
			int total=20;

			if (total>=clauses.size()) total=clauses.size()-1;

			System.err.print("Simplifying.");
			for (int p=0;p<total;p++) {
				//int val=simplify2(p);
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
		}*/

		toRemove = new BitSet(m_numAttribs);
		//coOccurrences();

		
		//Don't sort the clauses
		//java.util.Collections.sort(clauseList);
		
		/*for (int a=0;a<3;a++) {
			Clause c = clauseList.get(a);
			System.err.println("First clause is "+c);
			System.err.println("Set values = "+c.setValues);
			System.err.println(compare(c,clauseList.get(a+1)));
		}*/
		return clauseList;
	}

	public BitSet toRemove=null;

	/*public BitSet coOccurrences() {
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
	}*/

	/*public int simplify2(int index) {
		Clause o=null;

		try{
			o = clauseList.get(index);
		}
		catch (Exception e) {

		}
		//Clause may have been removed by previous pruning
		if (o!=null) return prune(check(o));
		else return -1; //0
	}*/
	ArrayList<Integer> keys;


	public boolean simplify=false;
	int startIndex=0;


	private void printClauses() {
		Iterator<Clause> it1 = clauses.values().iterator();

		while (it1.hasNext()) {
			System.err.println(it1.next());
		}
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double calculate(BitSet subset) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	public String getRevision() {
		// TODO Auto-generated method stub
		return "1.0";
	}
}




