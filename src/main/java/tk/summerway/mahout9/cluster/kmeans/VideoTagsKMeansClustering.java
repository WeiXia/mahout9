package tk.summerway.mahout9.cluster.kmeans;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.TanimotoDistanceMeasure;
import org.apache.mahout.math.hadoop.stats.BasicStats;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.HighDFWordsPruner;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VideoTagsKMeansClustering extends AbstractJob  {
    
    private static final Logger log = LoggerFactory.getLogger(VideoTagsKMeansClustering.class);

    public static String JOB_PATH = "video_tags_kmean_job";
    public static String ORIGINAL_DATA_PATH = "data";
    public static String VECTOR_PATH = "vectors";
    public static String CLUSTER_PATH = "clusters";
    public static String RESULT_DUMP_PATH = "dump";
    
    private Configuration conf = null;
    
    private void init() throws IOException {
        conf = getConf();
//        FileSystem fs = FileSystem.get(conf);
    }
    
    /**
     * read text file from hdfs and convert it to sequence file
     * key:Text, Value:StringTuple
     * 
     */
    // TODO
    private void text2seq() {
        
    }
    
    /**
     * calculate term frequency
     * @throws ClassNotFoundException 
     * @throws InterruptedException 
     * @throws IOException 
     */
    private void calculateTF() throws IOException, InterruptedException, ClassNotFoundException {
     // the minimum frequency of the feature in the entire corpus to be considered for inclusion in the sparse vector
        int minSupport = 5;
        
        // ngram paras
        int maxNGramSize = 1;
        int minLLRValue = 50;

        int reduceTasks = 1;
        int chunkSize = 200;
        
        boolean sequentialAccessOutput = true;
        boolean namedVectors = true;
        
        Path inputDir = new Path(JOB_PATH, ORIGINAL_DATA_PATH);
        Path outputDir = new Path(JOB_PATH, VECTOR_PATH);
        String tfDirName = "tf-vectors";
        DictionaryVectorizer.createTermFrequencyVectors(
                new Path(inputDir, "seqdata"),
                outputDir,
                tfDirName,
                conf,
                minSupport,
                maxNGramSize,
                minLLRValue,
                -1.0f,
                false,
                reduceTasks,
                chunkSize,
                sequentialAccessOutput,
                namedVectors);
    }
    
    private Pair<Long[], List<Path>> calculateDF() {
        // document frequency paras
        int minDf = 5;
        int maxDFPercent = 50;
        int maxDFSigma = -1;
        
        // calculate document frequency
        log.info("Calculating IDF");
        Pair<Long[], List<Path>> docFrequenciesFeatures = 
                TFIDFConverter.calculateDF(new Path(outputDir, tfDirName), 
                        outputDir, 
                        conf, 
                        chunkSize);
        return docFrequenciesFeatures;
    }
    
    private void pruneVectors(Pair<Long[], List<Path>> docFrequenciesFeatures) {
        int minDf = 5;
        int maxDFPercent = 50;
        int maxDFSigma = -1;
        int reduceTasks = 1;
        
        long maxDF = maxDFPercent;
        
        long vectorCount = docFrequenciesFeatures.getFirst()[1];
        if (maxDFSigma >= 0.0) {
          Path dfDir = new Path(outputDir, TFIDFConverter.WORDCOUNT_OUTPUT_FOLDER);
          Path stdCalcDir = new Path(outputDir, HighDFWordsPruner.STD_CALC_DIR);

          // Calculate the standard deviation
          double stdDev = BasicStats.stdDevForGivenMean(dfDir, stdCalcDir, 0.0, conf);
          maxDF = (int) (100.0 * maxDFSigma * stdDev / vectorCount);
        }

        long maxDFThreshold = (long) (vectorCount * (maxDF / 100.0f));
        
        // prune vectors
        HighDFWordsPruner.pruneVectors(tfDir,
                prunedTFDir,
                prunedPartialTFDir,
                maxDFThreshold,
                minDf,
                conf,
                docFrequenciesFeatures,
                -1.0f,
                false,
                reduceTasks);
        HadoopUtil.delete(new Configuration(conf), tfDir);
        
    }
    
    private void calculateTfIdf(Pair<Long[], List<Path>> docFrequenciesFeatures) {
        int minDf = 5;
        int maxDFPercent = 50;
        int norm = 2;
        boolean logNormalize = false;
        int reduceTasks = 1;
        boolean sequentialAccessOutput = true;
        boolean namedVectors = true;
        
        // calculate tf-idf weight
        TFIDFConverter.processTfIdf(
                new Path(outputDir, DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER),
                outputDir, 
                conf, 
                docFrequenciesFeatures, 
                minDf, 
                maxDFPercent, 
                norm, 
                logNormalize,
                sequentialAccessOutput, 
                namedVectors, 
                reduceTasks);
    }
    
    private void doClusteringJob() {
        Path vectorPath = new Path(VECTOR_PATH);
        Path vectorsFolder = new Path(vectorPath, "tfidf-vectors");
        Path canopyCentroids = new Path(vectorPath , "canopy-centroids");
        Path clusterOutput = new Path(vectorPath , "clusters");
        // using canopy to chose initail cluster
        CanopyDriver.run(vectorsFolder, 
                canopyCentroids,
                new EuclideanDistanceMeasure(), 
                250, 
                120, 
                false, 
                false);
        
        // run kmeans cluster
        KMeansDriver.run(conf, 
                vectorsFolder, 
                new Path(canopyCentroids, "clusters-0"),
                clusterOutput, 
                new TanimotoDistanceMeasure(), 
                0.01,
                20, 
                true, 
                false);
    }
    
    private void dumpResult() {

        // use clusterdump to dump clusters
        // -i clusters-x-final
        // -d dictionary.file-0
        // -dt sequencefile
        // -n 20
        
        // use seqdumper to dump videos and clusters
        // -i clusteredPoints
    }
    
    public static void main(String args[]) throws Exception {
        ToolRunner.run(new VideoTagsKMeansClustering(), args);
    }

    public int run(String[] arg0) throws Exception {
        init();
        text2seq();
        calculateTF();
        Pair<Long[], List<Path>> docFrequenciesFeatures = calculateDF();
        pruneVectors(docFrequenciesFeatures);
        calculateTfIdf(docFrequenciesFeatures);
        doClusteringJob();
        dumpResult();
        return 0;
    }
    
}
