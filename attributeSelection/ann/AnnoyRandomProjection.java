package weka.attributeSelection.ann;

import weka.core.Instances;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class AnnoyRandomProjection implements AnnInterface {

    private final int numTrees;
    private final int numFeatures;
    private final int maxLeafSize;
    private final List<Node> trees = new ArrayList<>();
    private double[][] data; // Store as primitive array for performance

    private static class Node {
        boolean isLeaf;
        int[] indices;
        double[] normal;
        double offset;
        Node left;
        Node right;
    }

    public AnnoyRandomProjection(int numTrees, int numFeatures, int maxLeafSize) {
        this.numTrees = numTrees;
        this.numFeatures = numFeatures;
        this.maxLeafSize = maxLeafSize;
    }

    public AnnoyRandomProjection(int numTrees, int numFeatures) {
        this(numTrees, numFeatures, 20);
    }
    
    @Override
    public void buildIndex(Instances instances) {
        this.data = AnnUtils.instancesToMatrix(instances);
        List<Integer> allIndices = IntStream.range(0, instances.numInstances()).boxed().collect(Collectors.toList());
        for (int i = 0; i < numTrees; i++) {
            trees.add(buildTree(allIndices));
        }
    }

    private Node buildTree(List<Integer> indices) {
        if (indices.size() <= maxLeafSize) {
            Node leaf = new Node();
            leaf.isLeaf = true;
            leaf.indices = indices.stream().mapToInt(i -> i).toArray();
            return leaf;
        }

        Random rand = new Random();
        double[] normal = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            normal[i] = rand.nextGaussian();
        }

        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();
        for (int idx : indices) {
            double dotProduct = 0;
            for (int i = 0; i < numFeatures; i++) {
                dotProduct += normal[i] * data[idx][i];
            }
            if (dotProduct > 0) rightIndices.add(idx);
            else leftIndices.add(idx);
        }

        if (leftIndices.isEmpty() || rightIndices.isEmpty()) {
            Node leaf = new Node();
            leaf.isLeaf = true;
            leaf.indices = indices.stream().mapToInt(i -> i).toArray();
            return leaf;
        }

        Node internalNode = new Node();
        internalNode.isLeaf = false;
        internalNode.normal = normal;
        internalNode.offset = 0;
        internalNode.left = buildTree(leftIndices);
        internalNode.right = buildTree(rightIndices);
        return internalNode;
    }

    @Override
    public int[] query(double[] vector, int k) {
        Set<Integer> candidates = new HashSet<>();
        for (Node tree : trees) {
            Node node = tree;
            while (!node.isLeaf) {
                double dotProduct = 0;
                for (int i = 0; i < numFeatures; i++) {
                    dotProduct += node.normal[i] * vector[i];
                }
                node = (dotProduct > node.offset) ? node.right : node.left;
            }
            for (int idx : node.indices) {
                candidates.add(idx);
            }
        }
        return candidates.stream().mapToInt(i -> i).toArray();
    }
}