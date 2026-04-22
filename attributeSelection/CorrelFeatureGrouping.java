/**
 * 
 */
package weka.attributeSelection;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.fuzzy.similarity.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Vector;

/**
 * @author rkj
 * 
 */
public class CorrelFeatureGrouping extends ASSearch implements OptionHandler {

	/** for serialization */
	static final long serialVersionUID = -6312951970168325471L;

	/** does the data have a class */
	protected boolean m_hasClass;
	
	/** avoid all group members of selected features appearing in the current reduct */
	protected boolean m_moreAvoids=true;

	protected class ClassCorrelation implements Serializable, Comparable<ClassCorrelation> {
		int index; //feature index
		double correlation=0;
		static final long serialVersionUID = -2930102837482622224L;

		public ClassCorrelation(int a, double c) {
			index=a;
			correlation=c;
		}

		@Override
		public int compareTo(ClassCorrelation other) {
			// TODO Auto-generated method stub
			return (int)(100000*(other.correlation-correlation));
		}

	}

	/** holds the class index */
	protected int m_classIndex;

	/** number of attributes in the data */
	protected int m_numAttribs;

	/** number of instances */
	protected int m_numInstances;

	//check at the end of hill-climbing if features can be pruned with no loss in the measure of subset goodness
	public boolean m_prune=false;

	/** Standard deviations of attributes (when using pearsons correlation) */
	private double[] m_std_devs;

	/**
	 * A threshold for determining groups
	 */
	protected double m_threshold=0.8;

	/*
	 * A cutoff threshold - any feature-feature correlations equal to or above this will be pruned
	 */
	protected double m_cutoff=1.0;

	//the AttributeEvaluator to use for ranking features within groups
	protected ASEvaluation m_ranker = new CorrelationAttributeEval();

	/** the merit of the best subset found */
	protected double m_bestMerit;

	/** the subset evaluation measure */
	protected ASEvaluation m_ASEval = new FuzzyRoughSubsetEval();

	protected Instances m_trainInstances;

	private StringBuffer searchOutput;

	/**
	 * 
	 */
	public CorrelFeatureGrouping() {
		m_threshold = 0.8;
		resetOptions();
	}

	double[] decisionCorr;	
	double[] sum;
	double[] sum_sq;
	double[] sqSum;


	/**
	 * Searches the attribute subset space by forward selection.
	 * 
	 * @param ASEval
	 *            the attribute evaluator to guide the search
	 * @param data
	 *            the training instances.
	 * @return an array (not necessarily ordered) of selected attribute indexes
	 * @throws Exception
	 *             if the search can't be completed
	 */
	public int[] search(ASEvaluation ASEval, Instances data) throws Exception {
		searchOutput = new StringBuffer("\n");
		m_numAttribs = data.numAttributes()-1;
		m_numInstances = data.numInstances();
		m_trainInstances = data;

		// if the data has no decision feature, m_classIndex is negative
		m_classIndex = data.classIndex();

		if (data != null) { // this is a fresh run so reset
			resetOptions();
			m_trainInstances = data;
		}

		m_ASEval = ASEval;
		m_ASEval.buildEvaluator(data);

		//Used to evaluate subsets in the hill-climbing search
		SubsetEvaluator evaluator =  (SubsetEvaluator)ASEval;

		//the final reduct (built up in hill-climbing search later on)
		BitSet reduct = new BitSet(m_numAttribs);

		// Array for the features in the dataset
		Relation connectedGraph = new Relation(m_numAttribs);

		//average is used as a threshold later for determining groups
		double average = 0; 
		double denom = 0;		
		decisionCorr = new double[m_numAttribs];	
		
		sum = new double[m_numAttribs];
		sum_sq = new double[m_numAttribs];
		sqSum = new double[m_numAttribs];
		m_std_devs = new double[m_numAttribs];

		//some initial calculations
		for (int k = 0; k < m_numAttribs; k++) {
			sum_sq[k]=0;
			sum[k]=0;

			for (int inst=0;inst<m_numInstances;inst++) {
				sum[k]+=data.instance(inst).value(k);
				sum_sq[k]+=(data.instance(inst).value(k)*data.instance(inst).value(k));
			}

			sqSum[k] = sum[k]*sum[k];
			m_std_devs[k] = 1.0;
		}

		double corrMax=-2, corrMin=2;

		System.err.print("Calculating correlations...");
		for (int k = 0; k < m_numAttribs; k++) {
			for (int j = k; j < m_numAttribs; j++) {
				double corr = Math.min(1,Math.abs(corr(k,j,data))); //was corr2(k,j,data)

				connectedGraph.setCell(k, j, corr);
				connectedGraph.setCell(j, k, corr);

				if (corr<corrMin) corrMin = corr;
				else if (corr>corrMax && k!=j) corrMax = corr;

				average+=corr;
				denom++;
			}
		}
		System.err.println(".done");

		//within-group rankings: relevancy to decision concept
		System.err.print("Generating within-group rankings...");
		ArrayList<ClassCorrelation> cc = new ArrayList<ClassCorrelation>(m_numAttribs);
		m_ranker.buildEvaluator(data);

		//the ranker needs to be an AttributeEvaluator
		if (!(m_ranker instanceof AttributeEvaluator)) {
			throw  new Exception(m_ranker.getClass().getName() + " is not an Attribute evaluator!");
		}

		AttributeEvaluator ranker = (AttributeEvaluator) m_ranker;

		for (int k=0;k<m_numAttribs;k++) {
			decisionCorr[k] = ranker.evaluateAttribute(k);
			cc.add(new ClassCorrelation(k,decisionCorr[k]));
		}

		System.err.println(".done");
		average/=denom;

		//The user can supply a different value (between 0 and 1) as a threshold, otherwise the midpoint between max correlation and the average is used.
		if (m_threshold<0||m_threshold>1) {
			//m_threshold = average+ ((corrMax-average)/2); //midpoint between the average correlation and the maximum correlation
			m_threshold = (0.85*corrMax);
		}

		searchOutput.append("Min correlation: "+corrMin+"\nMax correlation: "+corrMax+"\nAv = "+average);
		System.out.println("Min correlation: "+corrMin);
		System.out.println("Max correlation: "+corrMax);
		System.out.println("Av = "+average);
		System.out.println("Thresh = "+m_threshold);

		//sort the feature groups in descending order of relevancy to the decision feature
		Collections.sort(cc);
		Iterator<ClassCorrelation> it = cc.iterator();
		int[] attributeOrder = new int[m_numAttribs];
		int feat=0;

		while (it.hasNext()) {
			ClassCorrelation c = (it.next());
			attributeOrder[feat] = c.index;
			//System.err.println(feat+": "+c.index+" :"+c.correlation);
			feat++;
		}

		//Build groups, convert the groups into BitSets
		//- this makes the search stage quicker
		HashMap<Integer, BitSet> hmap = new HashMap<Integer, BitSet>();
		double averageGroupSize=0;
		BitSet alwaysAvoids = new BitSet(m_numAttribs); //always avoid these features (they are redundant)

		//loop through in order of relevancy, checking for redundancy 
		for (int i = 0; i < m_numAttribs; i++) {
			BitSet group = new BitSet(m_numAttribs);
			int k=attributeOrder[i];
			
			//if this feature hasn't been flagged already (by a more relevant feature)
			if (!alwaysAvoids.get(k)) {
				group.set(k);

				for (int j = 0; j < m_numAttribs; j++) {
					if (!alwaysAvoids.get(j)) {
						double val = connectedGraph.getCell(j, k);

						if (val>=m_threshold) {
							//if the correlation is above the cutoff point, then features k and j are redundant together
							if (val>=m_cutoff && k!=j) {
								alwaysAvoids.set(j);
							}
							else group.set(j); //add to the group
						}
					}
				}	
				averageGroupSize+=(double)group.cardinality();	
			}
			
			hmap.put(k, group);
		}

		System.err.println("Will remove "+alwaysAvoids.cardinality()+" feature(s)");

		averageGroupSize/=(double)(m_numAttribs - alwaysAvoids.cardinality());
		searchOutput.append("\nAverage group size: "+averageGroupSize+"\n\n");
		System.out.println("Average group size: "+averageGroupSize+"\n");

		//begin the hill-climbing phase
		boolean done = false;
		double current_best = -10000000;

		//work out the evaluation for the full dataset
		BitSet full = new BitSet(m_numAttribs);
		for (int a = 0; a < m_numAttribs; a++)
			if (a!=m_classIndex) full.set(a);

		//used for the stopping criterion
		double fullMeasure = evaluator.evaluateSubset(full);

		System.err.println("----------------------------");

		while (!done) {
			BitSet temp_group = (BitSet)reduct.clone();
			int best_attribute=-1;
			int iterations=0;

			//'avoids' keeps a list of features to avoid in this iteration
			BitSet avoids = (BitSet)alwaysAvoids.clone();
			//System.err.println("Avoids: "+avoids.cardinality());
			
			//hill-climbing search using the groups and rankings
			for (int f=0; f<m_numAttribs;f++) {
				int a = attributeOrder[f];

				//if we haven't selected this feature or flagged it before...
				if (!temp_group.get(a) && !avoids.get(a)) {
					int attr=a;
					iterations++;

					//if this feature is part of a group, get the group and choose a feature from it
					if (hmap.get(a).cardinality()>1) {
						BitSet group = (BitSet)(hmap.get(a)).clone();
						group.andNot(temp_group); //remove features that have already been selected at this point
						group.andNot(avoids); //don't consider group members that appear on the avoids list
						
						double maxCorr=-100000000;
						//go through features in the group and examine correlations with decision, and choose highest
						for (int i = group.nextSetBit(0); i >= 0; i = group.nextSetBit(i+1)) {	
							if (decisionCorr[i]>maxCorr) {
								maxCorr = decisionCorr[i];
								attr = i;
							}
						}			

						//avoid the other group members in this iteration
						avoids.or(group);
					}

					//evaluate the (temporary) addition of this feature
					temp_group.set(attr);
					double temp_merit = evaluator.evaluateSubset(temp_group);
					temp_group.clear(attr);

					if (temp_merit>=current_best) {
						current_best = temp_merit;
						best_attribute=attr;
					}

				}
			}

			//if no further improvement possible, break out of the loop
			if (best_attribute==-1) {
				done = false;
				break;
			}
			else {
				System.err.println("Iterations: "+iterations+" (full = "+(m_numAttribs-reduct.cardinality())+")");
				reduct.set(best_attribute);
				if (m_moreAvoids) {
					alwaysAvoids.or(hmap.get(best_attribute));					
				}
				searchOutput.append(reduct +" => "+current_best+"\n");
				System.err.println(reduct + " => "+current_best);
				if (current_best==fullMeasure) {
					done=false;
					break;
				}
			}
		}

		//try removing redundant features
		if (m_prune) {		
			for (int a = reduct.nextSetBit(0); a >= 0; a = reduct.nextSetBit(a + 1)) {
				reduct.clear(a);

				double val2 = evaluator.evaluateSubset(reduct);
				if (val2==fullMeasure&&val2!=0) { //prune this feature as it is redundant
					System.err.println("Pruned to: "+reduct+" => "+val2);
					searchOutput.append(reduct + " => "+val2+"\n");
				}
				else reduct.set(a);
			}
		}

		// Finally convert BitSet to array so it can be returned
		int[] subsetArray = new int[reduct.cardinality()];
		int y = 0;
		for (int z = reduct.nextSetBit(0); z >= 0; z=reduct.nextSetBit(z + 1)) {
			subsetArray[y] = z;
			y++;
		}

		return subsetArray;

	}

	//old version that doesn't handle nominal values
	private double corr2(int k, int j, Instances data) {
		double sum_prod=0;

		for (int inst=0;inst<m_numInstances;inst++) {
			sum_prod+= (data.instance(inst).value(k)*data.instance(inst).value(j));
		}

		double corr = ((double)m_numInstances)*sum_prod - (sum[k]*sum[j]);
		double denom1 = ((double)m_numInstances)*sum_sq[k] - sqSum[k];
		double denom2 = ((double)m_numInstances)*sum_sq[j] - sqSum[j];
		corr/= (Math.sqrt(denom1)*Math.sqrt(denom2));

		//corr goes between -1 and 1, so take absolute value
		return corr;
	}

	private double corr(int att1, int att2, Instances m_trainInstances) {
		boolean att1_is_num = (m_trainInstances.attribute(att1).isNumeric());
		boolean att2_is_num = (m_trainInstances.attribute(att2).isNumeric());

		if (att1_is_num && att2_is_num) {
			return  num_num(att1, att2);
		}
		else if (att2_is_num) {
			return  num_nom2(att1, att2);
		}
		else if (att1_is_num) {
			return  num_nom2(att2, att1);
		}

		return nom_nom(att1, att2);

	}


	private double num_num (int att1, int att2) {
		int i;
		Instance inst;
		double r, diff1, diff2, num = 0.0, sx = 0.0, sy = 0.0;
		double mx = m_trainInstances.meanOrMode(m_trainInstances.attribute(att1));
		double my = m_trainInstances.meanOrMode(m_trainInstances.attribute(att2));

		for (i = 0; i < m_numInstances; i++) {
			inst = m_trainInstances.instance(i);
			diff1 = (inst.isMissing(att1))? 0.0 : (inst.value(att1) - mx);
			diff2 = (inst.isMissing(att2))? 0.0 : (inst.value(att2) - my);
			num += (diff1*diff2);
			sx += (diff1*diff1);
			sy += (diff2*diff2);
		}

		if (sx != 0.0) {
			if (m_std_devs[att1] == 1.0) {
				m_std_devs[att1] = Math.sqrt((sx/m_numInstances));
			}
		}

		if (sy != 0.0) {
			if (m_std_devs[att2] == 1.0) {
				m_std_devs[att2] = Math.sqrt((sy/m_numInstances));
			}
		}

		if ((sx*sy) > 0.0) {
			r = (num/(Math.sqrt(sx*sy)));
			return  ((r < 0.0)? -r : r);
		}
		else {
			if (att1 != m_classIndex && att2 != m_classIndex) {
				return  1.0;
			}
			else {
				return  0.0;
			}
		}
	}

	private boolean m_missingSeparate=false;

	private double num_nom2 (int att1, int att2) {
		int i, ii, k;
		double temp;
		Instance inst;
		int mx = (int)m_trainInstances.
				meanOrMode(m_trainInstances.attribute(att1));
		double my = m_trainInstances.
				meanOrMode(m_trainInstances.attribute(att2));
		double stdv_num = 0.0;
		double diff1, diff2;
		double r = 0.0, rr;
		int nx = (!m_missingSeparate) 
				? m_trainInstances.attribute(att1).numValues() 
						: m_trainInstances.attribute(att1).numValues() + 1;

				double[] prior_nom = new double[nx];
				double[] stdvs_nom = new double[nx];
				double[] covs = new double[nx];

				for (i = 0; i < nx; i++) {
					stdvs_nom[i] = covs[i] = prior_nom[i] = 0.0;
				}

				// calculate frequencies (and means) of the values of the nominal 
				// attribute
				for (i = 0; i < m_numInstances; i++) {
					inst = m_trainInstances.instance(i);

					if (inst.isMissing(att1)) {
						if (!m_missingSeparate) {
							ii = mx;
						}
						else {
							ii = nx - 1;
						}
					}
					else {
						ii = (int)inst.value(att1);
					}

					// increment freq for nominal
					prior_nom[ii]++;
				}

				for (k = 0; k < m_numInstances; k++) {
					inst = m_trainInstances.instance(k);
					// std dev of numeric attribute
					diff2 = (inst.isMissing(att2))? 0.0 : (inst.value(att2) - my);
					stdv_num += (diff2*diff2);

					// 
					for (i = 0; i < nx; i++) {
						if (inst.isMissing(att1)) {
							if (!m_missingSeparate) {
								temp = (i == mx)? 1.0 : 0.0;
							}
							else {
								temp = (i == (nx - 1))? 1.0 : 0.0;
							}
						}
						else {
							temp = (i == inst.value(att1))? 1.0 : 0.0;
						}

						diff1 = (temp - (prior_nom[i]/m_numInstances));
						stdvs_nom[i] += (diff1*diff1);
						covs[i] += (diff1*diff2);
					}
				}

				// calculate weighted correlation
				for (i = 0, temp = 0.0; i < nx; i++) {
					// calculate the weighted variance of the nominal
					temp += ((prior_nom[i]/m_numInstances)*(stdvs_nom[i]/m_numInstances));

					if ((stdvs_nom[i]*stdv_num) > 0.0) {
						//System.out.println("Stdv :"+stdvs_nom[i]);
						rr = (covs[i]/(Math.sqrt(stdvs_nom[i]*stdv_num)));

						if (rr < 0.0) {
							rr = -rr;
						}

						r += ((prior_nom[i]/m_numInstances)*rr);
					}
					/* if there is zero variance for the numeric att at a specific 
         level of the catergorical att then if neither is the class then 
         make this correlation at this level maximally bad i.e. 1.0. 
         If either is the class then maximally bad correlation is 0.0 */
					else {if (att1 != m_classIndex && att2 != m_classIndex) {
						r += ((prior_nom[i]/m_numInstances)*1.0);
					}
					}
				}

				// set the standard deviations for these attributes if necessary
				// if ((att1 != classIndex) && (att2 != classIndex)) // =============
				if (temp != 0.0) {
					if (m_std_devs[att1] == 1.0) {
						m_std_devs[att1] = Math.sqrt(temp);
					}
				}

				if (stdv_num != 0.0) {
					if (m_std_devs[att2] == 1.0) {
						m_std_devs[att2] = Math.sqrt((stdv_num/m_numInstances));
					}
				}

				if (r == 0.0) {
					if (att1 != m_classIndex && att2 != m_classIndex) {
						r = 1.0;
					}
				}

				return  r;
	}


	private double nom_nom (int att1, int att2) {
		int i, j, ii, jj, z;
		double temp1, temp2;
		Instance inst;
		int mx = (int)m_trainInstances.
				meanOrMode(m_trainInstances.attribute(att1));
		int my = (int)m_trainInstances.
				meanOrMode(m_trainInstances.attribute(att2));
		double diff1, diff2;
		double r = 0.0, rr;
		int nx = (!m_missingSeparate) 
				? m_trainInstances.attribute(att1).numValues() 
						: m_trainInstances.attribute(att1).numValues() + 1;

				int ny = (!m_missingSeparate)
						? m_trainInstances.attribute(att2).numValues() 
								: m_trainInstances.attribute(att2).numValues() + 1;

						double[][] prior_nom = new double[nx][ny];
						double[] sumx = new double[nx];
						double[] sumy = new double[ny];
						double[] stdvsx = new double[nx];
						double[] stdvsy = new double[ny];
						double[][] covs = new double[nx][ny];

						for (i = 0; i < nx; i++) {
							sumx[i] = stdvsx[i] = 0.0;
						}

						for (j = 0; j < ny; j++) {
							sumy[j] = stdvsy[j] = 0.0;
						}

						for (i = 0; i < nx; i++) {
							for (j = 0; j < ny; j++) {
								covs[i][j] = prior_nom[i][j] = 0.0;
							}
						}

						// calculate frequencies (and means) of the values of the nominal 
						// attribute
						for (i = 0; i < m_numInstances; i++) {
							inst = m_trainInstances.instance(i);

							if (inst.isMissing(att1)) {
								if (!m_missingSeparate) {
									ii = mx;
								}
								else {
									ii = nx - 1;
								}
							}
							else {
								ii = (int)inst.value(att1);
							}

							if (inst.isMissing(att2)) {
								if (!m_missingSeparate) {
									jj = my;
								}
								else {
									jj = ny - 1;
								}
							}
							else {
								jj = (int)inst.value(att2);
							}

							// increment freq for nominal
							prior_nom[ii][jj]++;
							sumx[ii]++;
							sumy[jj]++;
						}

						for (z = 0; z < m_numInstances; z++) {
							inst = m_trainInstances.instance(z);

							for (j = 0; j < ny; j++) {
								if (inst.isMissing(att2)) {
									if (!m_missingSeparate) {
										temp2 = (j == my)? 1.0 : 0.0;
									}
									else {
										temp2 = (j == (ny - 1))? 1.0 : 0.0;
									}
								}
								else {
									temp2 = (j == inst.value(att2))? 1.0 : 0.0;
								}

								diff2 = (temp2 - (sumy[j]/m_numInstances));
								stdvsy[j] += (diff2*diff2);
							}

							// 
							for (i = 0; i < nx; i++) {
								if (inst.isMissing(att1)) {
									if (!m_missingSeparate) {
										temp1 = (i == mx)? 1.0 : 0.0;
									}
									else {
										temp1 = (i == (nx - 1))? 1.0 : 0.0;
									}
								}
								else {
									temp1 = (i == inst.value(att1))? 1.0 : 0.0;
								}

								diff1 = (temp1 - (sumx[i]/m_numInstances));
								stdvsx[i] += (diff1*diff1);

								for (j = 0; j < ny; j++) {
									if (inst.isMissing(att2)) {
										if (!m_missingSeparate) {
											temp2 = (j == my)? 1.0 : 0.0;
										}
										else {
											temp2 = (j == (ny - 1))? 1.0 : 0.0;
										}
									}
									else {
										temp2 = (j == inst.value(att2))? 1.0 : 0.0;
									}

									diff2 = (temp2 - (sumy[j]/m_numInstances));
									covs[i][j] += (diff1*diff2);
								}
							}
						}

						// calculate weighted correlation
						for (i = 0; i < nx; i++) {
							for (j = 0; j < ny; j++) {
								if ((stdvsx[i]*stdvsy[j]) > 0.0) {
									//System.out.println("Stdv :"+stdvs_nom[i]);
									rr = (covs[i][j]/(Math.sqrt(stdvsx[i]*stdvsy[j])));

									if (rr < 0.0) {
										rr = -rr;
									}

									r += ((prior_nom[i][j]/m_numInstances)*rr);
								}
								// if there is zero variance for either of the categorical atts then if
								// neither is the class then make this
								// correlation at this level maximally bad i.e. 1.0. If either is 
								// the class then maximally bad correlation is 0.0
								else {if (att1 != m_classIndex && att2 != m_classIndex) {
									r += ((prior_nom[i][j]/m_numInstances)*1.0);
								}
								}
							}
						}

						// calculate weighted standard deviations for these attributes
						// (if necessary)
						for (i = 0, temp1 = 0.0; i < nx; i++) {
							temp1 += ((sumx[i]/m_numInstances)*(stdvsx[i]/m_numInstances));
						}

						if (temp1 != 0.0) {
							if (m_std_devs[att1] == 1.0) {
								m_std_devs[att1] = Math.sqrt(temp1);
							}
						}

						for (j = 0, temp2 = 0.0; j < ny; j++) {
							temp2 += ((sumy[j]/m_numInstances)*(stdvsy[j]/m_numInstances));
						}

						if (temp2 != 0.0) {
							if (m_std_devs[att2] == 1.0) {
								m_std_devs[att2] = Math.sqrt(temp2);
							}
						}

						if (r == 0.0) {
							if (att1 != m_classIndex && att2 != m_classIndex) {
								r = 1.0;
							}
						}

						return  r;
	}

	public Enumeration<Option> listOptions() {
		Vector<Option> newVector = new Vector<Option>(4);
		newVector.addElement(new Option("\tSimilarity relation.", "R", 1,
				"-R <val>"));
		newVector.addElement(new Option("\tConnectives" + ".", "C", 1,
				"-C <val>"));
		newVector.addElement(new Option("\tComposition" + ".", "F", 1,
				"-F <val>"));
		newVector.addElement(new Option("\tkNN" + ".", "K", 1, "-K <val>"));
		return newVector.elements();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		resetOptions();
		String optionString;

		optionString = Utils.getOption('Z', options);
		if (optionString.length() != 0) {
			setThreshold(Double.valueOf(optionString));
		}

		else {
			setThreshold(1);
		}

		/*optionString = Utils.getOption('C', options);
		if (optionString.length() != 0) {
			setCutoff(Double.valueOf(optionString));
		}

		else {
			setCutoff(1);
		}*/

		optionString = Utils.getOption('S', options);
		if(optionString.length() != 0) {
			String nnSearchClassSpec[] = Utils.splitOptions(optionString);
			if(nnSearchClassSpec.length == 0) { 
				throw new Exception("Invalid AttributeEvaluator specification string."); 
			}
			String className = nnSearchClassSpec[0];
			nnSearchClassSpec[0] = "";

			setRanker( (ASEvaluation) Utils.forName( ASEvaluation.class, className, nnSearchClassSpec) );
		}
		else {
			setRanker(new CorrelationAttributeEval());
		}

		setPrune(Utils.getFlag('P', options));
		setMoreAvoids(Utils.getFlag('A', options));
	}

	/**
	 * Gets the current settings
	 * 
	 * @return an array of Strings suitable for passing to setOptions()
	 */
	public String[] getOptions() {
		Vector<String> result = new Vector<String>();

		result.add("-Z"); 
		result.add(""+getThreshold());

		result.add("-S"); 
		result.add(""+getRanker().getClass().getName());

		if (getPrune()) {
			result.add("-P");
		}

		if (getMoreAvoids()) {
			result.add("-A");
		}

		return result.toArray(new String[result.size()]);
	}

	public ASEvaluation getRanker() {
		return m_ranker;
	}

	public void setRanker(ASEvaluation as) {
		m_ranker=as;
	}

	public void setThreshold(double k) {
		m_threshold = k;
	}

	public double getThreshold() {
		return m_threshold;
	}

	public void setPrune(boolean p) {
		m_prune=p;
	}

	public boolean getPrune() {
		return m_prune;
	}


	public void setMoreAvoids(boolean p) {
		m_moreAvoids=p;
	}

	public boolean getMoreAvoids() {
		return m_moreAvoids;
	}
	
	/**
	 * Resets options
	 */
	protected void resetOptions() {
		m_ASEval = null;
		m_trainInstances = null;
	}

	public static void main(String[] args) {

	}

	/**
	 * returns a description of the search
	 * @return a description of the search as a String
	 */
	public String toString() {
		StringBuffer desc = new StringBuffer();
		desc.append("Hill-climber search using feature grouping\n");
		desc.append("and threshold: "+m_threshold);
		desc.append("\nUsing "+m_ranker.toString()+" to rank features within groups\n");

		return desc.toString()+searchOutput.toString();
	}

}
