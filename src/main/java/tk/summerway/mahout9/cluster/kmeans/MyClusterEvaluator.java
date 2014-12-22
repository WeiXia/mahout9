package tk.summerway.mahout9.cluster.kmeans;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Files;

public class MyClusterEvaluator {
    
    private static final Logger log = LoggerFactory
            .getLogger(MyClusterEvaluator.class);

    private String clusteredPointsFile = "";
    private String clustersFile = "";
    private String outputFile = "";
    private DistanceMeasure measure = null;
    private long maxPoints = 100L;
    
    private Map<Integer, Double> intraAvgDistances = null;
    private Map<Integer, Double> intraDensities = null;
    private Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints = null;
    private List<Cluster> clusters = null;
    private Map<ClusterPair, Double> clusterInterDistances = null;
    private double maxInterDistance = 0.0;
    private double minInterDistance = 0.0;
    private double avgInterDistance = 0.0;
    private double clusterInterDensity =0.0;
    
    public MyClusterEvaluator(String clusteredPointsPath, String clustersPath, 
            String outputFile, DistanceMeasure measure, long maxPoints) {
        this.clusteredPointsFile = clusteredPointsPath;
        this.clustersFile = clustersPath;
        this.outputFile = outputFile;
        this.measure = measure;
        this.maxPoints = maxPoints;
    }
        
    public void evaluateClusters(Configuration conf) throws Exception {
        // calculate intra distances and densities
        calcIntraDistances(conf);
        
        // calculate inter distances
        calcInterDistances(conf);
        
        // evaluate clusters

        // print result
        printResult();
        
    }
    
    private void calcIntraDistances(Configuration conf) throws Exception {
        Path pointsPath = new Path(clusteredPointsFile);
        log.info("Points Input Path: " + pointsPath);

        intraAvgDistances = Maps.newHashMap();
        intraDensities = Maps.newHashMap();
        clusterIdToPoints = ClusterDumper.readPoints(pointsPath, maxPoints, conf);
        
        for (Integer clusterId : clusterIdToPoints.keySet()) {
            List<WeightedPropertyVectorWritable> points = clusterIdToPoints
                    .get(clusterId);
            double max = 0;
            double min = Double.MAX_VALUE;
            double sum = 0;
            int count = 0;
            for (int i = 0; i < points.size(); i++) {
                for (int j = i + 1; j < points.size(); j++) {
                    Vector ipoint = points.get(i).getVector();
                    Vector jpoint = points.get(j).getVector();
                    double d = measure.distance(ipoint, jpoint);
                    min = Math.min(d, min);
                    max = Math.max(d, max);
                    sum += d;
                    count++;
                }
            }
            double avgIntraDistance = sum / count;
            double density = (sum / count - min) / (max - min);
            intraAvgDistances.put(clusterId, avgIntraDistance);
            intraDensities.put(clusterId, density);
            log.info("ClusterID : " + clusterId + " Intra Average Distance : "
                    + avgIntraDistance + " Intra Density : " + density);
        }
    }
    
    private void calcInterDistances(Configuration conf) throws Exception {
        Path clustersPath = new Path(clustersFile);
        log.info("Clusters Input Path: " + clustersPath);

        clusters = loadClusters(conf, clustersPath);
        clusterInterDistances = Maps.newHashMap();

        double max = 0;
        double min = Double.MAX_VALUE;
        double sum = 0;
        int count = 0;
        for (int i = 0; i < clusters.size(); i++) {
            for (int j = i + 1; j < clusters.size(); j++) {
                Cluster clusterI = clusters.get(i);
                Cluster clusterJ = clusters.get(j);
                double d = measure.distance(clusterI.getCenter(),
                        clusterJ.getCenter());
                ClusterPair cp = new ClusterPair(clusterI.getId(), clusterJ.getId());
                clusterInterDistances.put(cp, d);
                min = Math.min(d, min);
                max = Math.max(d, max);
                sum += d;
                count++;
            }
        }
        maxInterDistance = max;
        minInterDistance = min;
        avgInterDistance = sum / count;
        clusterInterDensity = (sum / count - min) / (max - min);
        log.info("Maximum Intercluster Distance: " + maxInterDistance);
        log.info("Minimum Intercluster Distance: " + minInterDistance);
        log.info("Average Intercluster Distance: " + avgInterDistance);
        log.info("Scaled Inter-Cluster Density: " + clusterInterDensity);
    }
    
    private static List<Cluster> loadClusters(Configuration conf,
            Path clustersIn) {
        List<Cluster> clusters = Lists.newArrayList();
        for (ClusterWritable clusterWritable : new SequenceFileDirValueIterable<ClusterWritable>(
                clustersIn, PathType.LIST, PathFilters.logsCRCFilter(), conf)) {
            Cluster cluster = clusterWritable.getValue();
            clusters.add(cluster);
        }
        return clusters;
    }
    
    private void printResult() throws IOException {
        Writer writer = null;
        try {
            writer = Files.newWriter(new File(outputFile), Charsets.UTF_8);
            writer.append("Cluster intra info :");
            writer.append("\n");
            for (Integer clusterId : intraAvgDistances.keySet()) {
                writer.append("ClusterID : " + clusterId
                        + " Intra Average Distance : "
                        + intraAvgDistances.get(clusterId) + " Intra Density : "
                        + intraDensities.get(clusterId));
                writer.append("\n");
            }
            writer.append("Cluster inter info :");
            writer.append("\n");
            writer.append("Maximum Intercluster Distance: " + maxInterDistance);
            writer.append("\n");
            writer.append("Minimum Intercluster Distance: " + minInterDistance);
            writer.append("\n");
            writer.append("Average Intercluster Distance: " + avgInterDistance);
            writer.append("\n");
            writer.append("Scaled Inter-Cluster Density: " + clusterInterDensity);
            writer.append("\n");
            for (ClusterPair clusterPair : clusterInterDistances.keySet()) {
                writer.append("The distance between cluster : "
                        + clusterPair.clusterId1 + " and cluster : "
                        + clusterPair.clusterId2 + " is "
                        + clusterInterDistances.get(clusterPair));
                writer.append("\n");
            }
            writer.flush();
        } catch (Exception e) {
            log.error("Error occured in print cluster's intra distance info!", e);
        } finally {
            writer.close();
        }
    }
    
    private class ClusterPair {
        
        private int clusterId1 = 0;
        private int clusterId2 = 0;
        
        public ClusterPair(int clusterId1, int clusterId2) {
            this.clusterId1 = clusterId1;
            this.clusterId2 = clusterId2;
        }
        
        @Override
        public int hashCode() {
            return new HashCodeBuilder().
                append(clusterId1 < clusterId2 ? clusterId1 : clusterId2).
                append(clusterId1 < clusterId2 ? clusterId2 : clusterId1).
                toHashCode();
        }
        
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof ClusterPair))
                return false;
            if (obj == this)
                return true;
            ClusterPair objCP = (ClusterPair)obj;
            if (objCP.clusterId1 == this.clusterId1 && objCP.clusterId2 == this.clusterId2) {
                return true;
            } else if (objCP.clusterId1 == this.clusterId2 && objCP.clusterId1 == this.clusterId2) {
                return true;
            } else {
                return false;
            }
        }
        
        @Override
        public String toString() {
            return "ClusterId1:" + String.valueOf(clusterId1)
                    + " ClusterId2:" + String.valueOf(clusterId2); 
        }
        
    }
    
    public static void main(String args[]) throws Exception {
        MyClusterEvaluator ce = new MyClusterEvaluator("", "", "", null, 0);
        ClusterPair cp1  = ce.new ClusterPair(11, 22);
        ClusterPair cp2  = ce.new ClusterPair(22, 11);
        ClusterPair cp3  = ce.new ClusterPair(11, 33);
        Map<ClusterPair, Double> clusterInterDistances = Maps.newHashMap();
        clusterInterDistances.put(cp1, 1.0);
        clusterInterDistances.put(cp2, 2.0);
        clusterInterDistances.put(cp3, 3.0);
        for (ClusterPair cp : clusterInterDistances.keySet()) {
            System.out.println(cp + " " + clusterInterDistances.get(cp));
        }
        
    }
    
}