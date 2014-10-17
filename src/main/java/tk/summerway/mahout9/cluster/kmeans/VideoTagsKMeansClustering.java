package tk.summerway.mahout9.cluster.kmeans;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.hadoop.stats.BasicStats;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.HighDFWordsPruner;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * directory structure
 * video_tags_kmean_job
 * video_tags_kmean_job/data
 * video_tags_kmean_job/vectors
 * video_tags_kmean_job/clusters
 * video_tags_kmean_job/dump
 * 
 * @author xiawei
 *
 */
public class VideoTagsKMeansClustering extends AbstractJob  {
    
    private static final Logger log = LoggerFactory.getLogger(VideoTagsKMeansClustering.class);

    /**
     * Job home path
     */
    public static String JOB_PATH = "video_tags_kmean_job";
    /**
     * Job orginal text and sequenced data
     */
    public static String ORIGINAL_DATA_PATH = "data";
    /**
     * vectors data. Include tf-vectors, df-vectors, tf-vectors-pruned, tfidf-vectors
     */
    public static String VECTOR_PATH = "vectors";
    /**
     * cluster data
     */
    public static String CLUSTER_PATH = "clusters";
    /**
     * dump data
     */
    public static String RESULT_DUMP_PATH = "dump";
    
    private Configuration conf = null;
    
    // the minimum frequency of the feature in the entire corpus to be considered for inclusion in the sparse vector
    private int minSupport = 5;
    
    // ngram paras
    private int maxNGramSize = 1;
    private int minLLRValue = 50;

    // document frequency paras
    private int minDf = 5;
    private int maxDFPercent = 50;
    private int maxDFSigma = -1;
    
    // normalize para
    private int norm = 2;
    private boolean logNormalize = false;
    
    private int reduceTasks = 1;
    private int chunkSize = 200;
    
    private boolean sequentialAccessOutput = true;
    private boolean namedVectors = true;
    
    // kmeans para
    private int k = 5;
    private double convergenceDelta = 0.01;
    private int maxIterations = 10;
    
    private void init() throws IOException {
        conf = getConf();
//        FileSystem fs = FileSystem.get(conf);
    }
    
    
    /**
     * read text file from hdfs and convert it to sequence file
     * input data format : video_id tag1 tag2 ...... tagN
     * output data format : key-Text, Value-StringTuple
     * @throws IOException 
     */
    private void text2seq(String localTextFile) throws IOException {
        log.info("local file path : " + localTextFile);
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        
        // local text file
        File file = new File(localTextFile);
        
        // output seq file
        Path originalDataDir = new Path(JOB_PATH, ORIGINAL_DATA_PATH);
        Path seqData = new Path(originalDataDir, "seqdata");
        // delete old seq data
        HadoopUtil.delete(conf, seqData);
        
        // prepare local file reader
        BufferedReader reader = null;
        reader = new BufferedReader(new FileReader(file));
        // prepare seq file writer, this will create a empty file
        SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
                seqData, Text.class, StringTuple.class);
        
        String tempString = null;
//        int line = 1;
        log.info("wiriting data to seq file : " + seqData);
        while ((tempString = reader.readLine()) != null) {
//            System.out.println("line " + line + ": " + tempString);
//            line++;
            String[] data = tempString.split(" ");
            String videoId = data[0];
            StringTuple tags = new StringTuple();
            for (int i = 1; i < data.length - 1; i++) {
                tags.add(data[i]);
            }
            writer.append(new Text(videoId), tags);
        }
        reader.close();
        writer.close();
        log.info("seq file writen");
    }
    
    /**
     * calculate term frequency
     * @throws ClassNotFoundException 
     * @throws InterruptedException 
     * @throws IOException 
     */
    private void calculateTF() throws IOException, InterruptedException, ClassNotFoundException {
        // original data path
        Path originalDataDir = new Path(JOB_PATH, ORIGINAL_DATA_PATH);
        // vector data path
        Path vectorDir = new Path(JOB_PATH, VECTOR_PATH);
        String tfDirName = "tf-vectors";
        
        // delete old data
        HadoopUtil.delete(conf, vectorDir);
        
        DictionaryVectorizer.createTermFrequencyVectors(
                new Path(originalDataDir, "seqdata"),
                vectorDir,
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
    
    private Pair<Long[], List<Path>> calculateDF()
            throws IOException, InterruptedException, ClassNotFoundException {
        // vector data path
        Path vectorDir = new Path(JOB_PATH, VECTOR_PATH);
        Path tfVectorDir = new Path(vectorDir, "tf-vectors");
                
        // document frequency vector path
        String dfDirName = "df-vectors";
        Path dfDir = new Path(vectorDir, dfDirName);
        
        // delete old data
        HadoopUtil.delete(conf, dfDir);
        
        // calculate document frequency
        log.info("Calculating IDF");
        Pair<Long[], List<Path>> docFrequenciesFeatures = 
                TFIDFConverter.calculateDF(tfVectorDir, 
                        dfDir, 
                        conf, 
                        chunkSize);
        return docFrequenciesFeatures;
    }

    private void pruneVectors(Pair<Long[], List<Path>> docFrequenciesFeatures)
            throws IOException, InterruptedException, ClassNotFoundException {
        // vector data path
        Path vectorDir = new Path(JOB_PATH, VECTOR_PATH);
        // term frequncy vector path
        String tfVectorDirName = "tf-vectors";
        Path tfVectorDir = new Path(vectorDir, tfVectorDirName);
        // pruned vector path
        String prunedTfVectorDirName = "tf-vectors-pruned";
        Path prunedTFVectorDir = new Path(vectorDir, prunedTfVectorDirName);
        Path prunedPartialTFDir =
                new Path(vectorDir, DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER + "-partial");

        // delete old data
        HadoopUtil.delete(conf, prunedTFVectorDir);
        HadoopUtil.delete(conf, prunedPartialTFDir);
        
        // calculate threshold
        // TODO to be understand
        long maxDF = maxDFPercent;
        long vectorCount = docFrequenciesFeatures.getFirst()[1];
        if (maxDFSigma >= 0.0) {
          Path dfDir = new Path(vectorDir, TFIDFConverter.WORDCOUNT_OUTPUT_FOLDER);
          Path stdCalcDir = new Path(vectorDir, HighDFWordsPruner.STD_CALC_DIR);

          // Calculate the standard deviation
          double stdDev = BasicStats.stdDevForGivenMean(dfDir, stdCalcDir, 0.0, conf);
          maxDF = (int) (100.0 * maxDFSigma * stdDev / vectorCount);
        }
        long maxDFThreshold = (long) (vectorCount * (maxDF / 100.0f));
        
        // prune the term frequency vectors
        HighDFWordsPruner.pruneVectors(tfVectorDir,
                prunedTFVectorDir,
                prunedPartialTFDir,
                maxDFThreshold,
                minDf,
                conf,
                docFrequenciesFeatures,
                -1.0f,
                false,
                reduceTasks);
    }
    
    private void calculateTfIdf(Pair<Long[], List<Path>> docFrequenciesFeatures)
            throws IOException, InterruptedException, ClassNotFoundException {
        // vector data path
        Path vectorDir = new Path(JOB_PATH, VECTOR_PATH);
        // pruned tf-vector path
        String prunedTfVectorDirName = "tf-vectors-pruned";
        Path prunedTFVectorDir = new Path(vectorDir, prunedTfVectorDirName);
        // tf-idf weighted vector path
        String tfidfVectorDirName = "tfidf-vectors";
        Path tfidfVectorDir = new Path(vectorDir, tfidfVectorDirName);
        
        // delete old data
        HadoopUtil.delete(conf, tfidfVectorDir);
        
        // calculate tf-idf weight
        TFIDFConverter.processTfIdf(
                prunedTFVectorDir,
                tfidfVectorDir, 
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
    
    private void doClusteringJob() throws IOException, InterruptedException, ClassNotFoundException {
        // vector data path
        Path vectorDir = new Path(JOB_PATH, VECTOR_PATH);
        // pruned tfidf-vector path
        String tfidfVectorDirName = "tfidf-vectors";
        Path tfidfVectorDir = new Path(vectorDir, tfidfVectorDirName);
        
        // cluster data path
        Path clusterPath = new Path(JOB_PATH, CLUSTER_PATH);
        
//        Path canopyCentroids = new Path(vectorPath , "canopy-centroids");
//        // using canopy to chose initail cluster
//        CanopyDriver.run(conf,
//                vectorsFolder, 
//                canopyCentroids,
//                new EuclideanDistanceMeasure(), 
//                250, 
//                120, 
//                false,
//                0,
//                false);
        
        // run kmeans cluster
//        Path directoryContainingConvertedInput = new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT);
//        log.info("Preparing Input");
//        InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
        log.info("Running random seed to get initial clusters");
        Path initCluster = new Path(clusterPath, "random-seeds");
        initCluster = RandomSeedGenerator.buildRandom(conf, 
                tfidfVectorDir, 
                initCluster, 
                k, 
                new EuclideanDistanceMeasure());
        
        log.info("Running KMeans with k = {}", k);
        KMeansDriver.run(conf, 
                tfidfVectorDir, 
                initCluster, 
                clusterPath, 
                convergenceDelta,
                maxIterations, 
                true, 
                0.0, 
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

    public int run(String[] args) throws Exception {
        init();
        text2seq(args[0]);
        calculateTF();
//        Pair<Long[], List<Path>> docFrequenciesFeatures = calculateDF();
//        pruneVectors(docFrequenciesFeatures);
//        calculateTfIdf(docFrequenciesFeatures);
//        doClusteringJob();
//        dumpResult();
        return 0;
    }
    
}
