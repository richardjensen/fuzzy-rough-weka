package weka.attributeSelection.ann;

import weka.core.Instances;
import java.util.*;

public class LSH implements AnnInterface {
    private final int b;
    private final int r;
    private final int numHashes;
    private final int numFeatures;
    private final double bucketWidth;
    private final double[][] planes;
    private final double[] offsets;
    private final Map<String, List<Integer>>[] hashTables;
    private double[][] data;

    @SuppressWarnings("unchecked")
    public LSH(int b, int r, int numFeatures, double bucketWidth) {
        this.b = b;
        this.r = r;
        this.numHashes = b * r;
        this.numFeatures = numFeatures;
        this.bucketWidth = bucketWidth;

        Random rand = new Random();
        this.planes = new double[numHashes][numFeatures];
        for (int i = 0; i < numHashes; i++) {
            for (int j = 0; j < numFeatures; j++) {
                planes[i][j] = rand.nextGaussian();
            }
        }
        this.offsets = new double[numHashes];
        for (int i = 0; i < numHashes; i++) {
            offsets[i] = rand.nextDouble() * bucketWidth;
        }
        this.hashTables = new HashMap[b];
        for (int i = 0; i < b; i++) {
            hashTables[i] = new HashMap<>();
        }
    }
    
    public LSH(int b, int r, int numFeatures) {
        this(b, r, numFeatures, 4.0);
    }

    private int[] hash(double[] vec) {
        int[] signature = new int[numHashes];
        for (int i = 0; i < numHashes; i++) {
            double dotProduct = 0;
            for (int j = 0; j < numFeatures; j++) {
                dotProduct += planes[i][j] * vec[j];
            }
            signature[i] = (int) Math.floor((dotProduct + offsets[i]) / bucketWidth);
        }
        return signature;
    }

    @Override
    public void buildIndex(Instances instances) {
        this.data = AnnUtils.instancesToMatrix(instances);
        for (int i = 0; i < data.length; i++) {
            int[] signature = hash(data[i]);
            for (int bandIdx = 0; bandIdx < b; bandIdx++) {
                StringBuilder bandKeyBuilder = new StringBuilder();
                for (int rowIdx = 0; rowIdx < r; rowIdx++) {
                    bandKeyBuilder.append(signature[bandIdx * r + rowIdx]).append(',');
                }
                String bandKey = bandKeyBuilder.toString();
                hashTables[bandIdx].computeIfAbsent(bandKey, k -> new ArrayList<>()).add(i);
            }
        }
    }

    @Override
    public int[] query(double[] vector, int k) {
        Set<Integer> candidates = new HashSet<>();
        int[] signature = hash(vector);
        for (int bandIdx = 0; bandIdx < b; bandIdx++) {
            StringBuilder bandKeyBuilder = new StringBuilder();
            for (int rowIdx = 0; rowIdx < r; rowIdx++) {
                bandKeyBuilder.append(signature[bandIdx * r + rowIdx]).append(',');
            }
            String bandKey = bandKeyBuilder.toString();
            if (hashTables[bandIdx].containsKey(bandKey)) {
                candidates.addAll(hashTables[bandIdx].get(bandKey));
            }
        }
        return candidates.stream().mapToInt(i -> i).toArray();
    }
}