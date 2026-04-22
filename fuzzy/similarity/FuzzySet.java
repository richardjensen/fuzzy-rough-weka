package weka.fuzzy.similarity;

import java.io.Serializable;

public final class FuzzySet implements Serializable {

static final long serialVersionUID = 747878400815211184L;

	boolean discrete;
	boolean empty = false;
	int i = 0;// column number
	int TYPE = -1;

	// experimental: alpha cuts
	static double alphaCut = 0.7;

	// continuous
	public static final int TRIANGULAR = 1;
	public static final int TRAPEZOIDAL = 2;
	public static final int L_SHOULDERED = 3;
	public static final int R_SHOULDERED = 4;
	public static final int GAUSSIAN = 5;
	public static final int INDIVIDUAL = 6;

	public static int CONNECT = 2;
	public static int COMPOSE = 2;

	public static final int MINMAX = 1;
	public static final int LIMITED = 2;// Lukasiewicz
	public static final int ALGEBRAIC = 3;
	public static final int DRASTIC = 4;
	public static final int TM = 5;

	String name = "";
	double a, b, c, d;
	double m1, m2;
	double scale = 1;
	static double result = 0;

	public FuzzySet(String n, int num, double a1, double b1, double c1, double d1,
			int type) {
		i = num;
		discrete = false;
		name = n;
		TYPE = type;
		a = a1;
		b = b1;
		c = c1;
		d = d1;
	}

	public FuzzySet(String na) {
		discrete = true;
		name = na;
	}


	public final double getMembership(double object) {
		return membership(object);
	}


	public final double getMembership(long object) {
		return membership(object);
	}

	public final double membership(double index) {
		double mem = 0;

		switch (TYPE) {
		case TRIANGULAR:
			mem = triangular(index);
			break;

		case TRAPEZOIDAL:
			mem = trapezoidal(index);
			break;

		case L_SHOULDERED:
			mem = leftShouldered(index);
			break;

		case R_SHOULDERED:
			mem = rightShouldered(index);
			break;

		case GAUSSIAN:
			mem = gaussian(index);
			break;

		case INDIVIDUAL:
			mem = individual(index);
			break;
		}
		return mem;
	}

	public final double triangular(double x) {
		double mem = 0;
		// mem = Math.max(Math.min((x-a)/(b-a),(c-x)/(c-b)),0);
		m1 = (x - a) / (b - a);
		m2 = (c - x) / (c - b);
		if (m1 < m2)
			mem = m1;
		else
			mem = m2;
		if (mem < 0)
			mem = 0;
		return mem;
	}

	public final double trapezoidal(double x) {
		double mem = 0;
		mem = Math.max(Math.min(Math.min((x - a) / (b - a), 1), (d - x)
				/ (d - c)), 0);
		return mem;
	}

	public final double leftShouldered(double x) {
		double mem = 0;
		if (x <= a)
			mem = 1;
		else if ((a < x) && (x < b))
			mem = 1 - ((x - a) / (b - a));
		else if (x >= b)
			mem = 0f;
		return mem;
	}

	public final double rightShouldered(double x) {
		double mem = 0;
		if (x <= a)
			mem = 0;
		else if ((a < x) && (x < b))
			mem = ((x - a) / (b - a));
		else if (x >= b)
			mem = 1;
		return mem;
	}

	// parameter a is mean, b is variance
	public final double gaussian(double x) {
		// d == sigma
		double val = -0.5 * ((a - x) * (a - x)) / (b);
		return (double) Math.exp(val);
	}

	public final double individual(double x) {
		return a;
	}

	public final FuzzySet clone1() {
		return new FuzzySet(name, i, a, b, c, d, TYPE);
	}

	public final String toString() {
		String ret = "";

		if (discrete) {
			// for (int i=0;i<values.size();i++)
			// ret+=String.valueOf(values.get(i))+",";
		} else {
			ret = name;
		}
		return ret;
	}

	public boolean separate(FuzzySet fs) {
		return ((TYPE == L_SHOULDERED && fs.TYPE == R_SHOULDERED) || (TYPE == R_SHOULDERED && fs.TYPE == L_SHOULDERED));
	}
}
