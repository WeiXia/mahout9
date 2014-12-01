package tk.summerway.mahout9.cluster.kmeans;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stats.BasicStats;
import org.apache.mahout.utils.SequenceFileDumper;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.HighDFWordsPruner;
import org.apache.mahout.vectorizer.collocations.llr.LLRReducer;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import tk.summerway.mahout9.tools.MyClusterDumper;
import tk.summerway.mahout9.tools.RandomPointsUtil;

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
    
    private static final String OP_TYPE_ONLY_VECTORIZE = "1";
    
    private static final String OP_TYPE_ONLY_CLUSTERING = "2";
    
    private static final String OP_TYPE_ONLY_DUMP = "3";
    
    private String opType = null;
    
    private String inputDir = null;
    
    // the minimum tag count that video or ugc member has
    private static int MIN_TAG_COUNT = 50;
    
    // the minimum frequency of the feature in the entire corpus to be considered for inclusion in the sparse vector
    private int minSupport = 2;
    
    // ngram paras
    private int maxNGramSize = 1;
    private float minLLRValue = LLRReducer.DEFAULT_MIN_LLR;

    // document frequency paras
    private int minDf = 1;
    private int maxDFPercent = 99;
    private double maxDFSigma = -1.0;
    
    // normalize para
    private float norm = PartialVectorMerger.NO_NORMALIZING;
    private boolean logNormalize = false;
    
    private int reduceTasks = 1;
    private int chunkSize = 100;
    
    private boolean sequentialAccessOutput = false;
    private boolean namedVectors = false;
    
    // kmeans para
    private int kValue = -1;
    private double convergenceDelta = 0.5;
    private int maxIterations = 1;
    private DistanceMeasure measure = null;
    
    private boolean init(String[] args) throws IOException {
        if (!buildParse(args)) {
            return false;
        }
        conf = getConf();
        return true;
    }
    
    private boolean buildParse(String[] args) {
        DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
        ArgumentBuilder abuilder = new ArgumentBuilder();
        GroupBuilder gbuilder = new GroupBuilder();
        
        Option opTypeOpt = obuilder
                .withLongName("opType")
                .withRequired(false)
                .withArgument(
                        abuilder.withName("opType").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription(
                        "1 prepare vectors; 2 run kmeans job; 3 dump cluster and points; default do all things together")
                .withShortName("ot").create();

        Option inputDirOpt = DefaultOptionCreator.inputOption().create();

        Option minSupportOpt = obuilder
                .withLongName("minSupport")
                .withArgument(
                        abuilder.withName("minSupport").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription("(Optional) Minimum Support. Default Value: 2")
                .withShortName("s").create();

        // Option analyzerNameOpt =
        // obuilder.withLongName("analyzerName").withArgument(
        // abuilder.withName("analyzerName").withMinimum(1).withMaximum(1).create()).withDescription(
        // "The class name of the analyzer").withShortName("a").create();

        Option chunkSizeOpt = obuilder
                .withLongName("chunkSize")
                .withArgument(
                        abuilder.withName("chunkSize").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription(
                        "The chunkSize in MegaBytes. Default Value: 100MB")
                .withShortName("chunk").create();

        Option minDFOpt = obuilder
                .withLongName("minDF")
                .withRequired(false)
                .withArgument(
                        abuilder.withName("minDF").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription(
                        "The minimum document frequency.  Default is 1")
                .withShortName("md").create();

        Option maxDFPercentOpt = obuilder
                .withLongName("maxDFPercent")
                .withRequired(false)
                .withArgument(
                        abuilder.withName("maxDFPercent").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription(
                        "The max percentage of docs for the DF.  Can be used to remove really high frequency terms."
                                + " Expressed as an integer between 0 and 100. Default is 99.  If maxDFSigma is also set, "
                                + "it will override this value.")
                .withShortName("x").create();

        Option maxDFSigmaOpt = obuilder
                .withLongName("maxDFSigma")
                .withRequired(false)
                .withArgument(
                        abuilder.withName("maxDFSigma").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription(
                        "What portion of the tf (tf-idf) vectors to be used, expressed in times the standard deviation (sigma) "
                                + "of the document frequencies of these vectors. Can be used to remove really high frequency terms."
                                + " Expressed as a double value. Good value to be specified is 3.0. In case the value is less "
                                + "than 0 no vectors will be filtered out. Default is -1.0.  Overrides maxDFPercent")
                .withShortName("xs").create();

        Option minLLROpt = obuilder
                .withLongName("minLLR")
                .withRequired(false)
                .withArgument(
                        abuilder.withName("minLLR").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription(
                        "(Optional)The minimum Log Likelihood Ratio(Float)  Default is "
                                + LLRReducer.DEFAULT_MIN_LLR)
                .withShortName("ml").create();

        Option numReduceTasksOpt = obuilder
                .withLongName("numReducers")
                .withArgument(
                        abuilder.withName("numReducers").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription(
                        "(Optional) Number of reduce tasks. Default Value: 1")
                .withShortName("nr").create();

        Option powerOpt = obuilder
                .withLongName("norm")
                .withRequired(false)
                .withArgument(
                        abuilder.withName("norm").withMinimum(1).withMaximum(1)
                                .create())
                .withDescription(
                        "The norm to use, expressed as either a float or \"INF\" if you want to use the Infinite norm.  "
                                + "Must be greater or equal to 0.  The default is not to normalize")
                .withShortName("n").create();

        Option logNormalizeOpt = obuilder
                .withLongName("logNormalize")
                .withRequired(false)
                .withDescription(
                        "(Optional) Whether output vectors should be logNormalize. If set true else false")
                .withShortName("lnorm").create();

        Option maxNGramSizeOpt = obuilder
                .withLongName("maxNGramSize")
                .withRequired(false)
                .withArgument(
                        abuilder.withName("ngramSize").withMinimum(1)
                                .withMaximum(1).create())
                .withDescription(
                        "(Optional) The maximum size of ngrams to create"
                                + " (2 = bigrams, 3 = trigrams, etc) Default Value:1")
                .withShortName("ng").create();

        Option sequentialAccessVectorOpt = obuilder
                .withLongName("sequentialAccessVector")
                .withRequired(false)
                .withDescription(
                        "(Optional) Whether output vectors should be SequentialAccessVectors. If set true else false")
                .withShortName("seq").create();

        Option namedVectorOpt = obuilder
                .withLongName("namedVector")
                .withRequired(false)
                .withDescription(
                        "(Optional) Whether output vectors should be NamedVectors. If set true else false")
                .withShortName("nv").create();
        
        Option helpOpt = obuilder.withLongName("help")
                .withDescription("Print out help").withShortName("h").create();
        
        Option kValueOpt = obuilder
                .withLongName("kValue")
                .withArgument(abuilder.withName("kValue").withMinimum(1).withMaximum(1).create())
                .withDescription("k-means k value")
                .withShortName("k").create();
        
        Option convergenceDeltaOpt = obuilder
                .withLongName("convergenceDelta")
                .withArgument(abuilder.withName("convergenceDelta").withMinimum(1).withMaximum(1).create())
                .withDescription("k-means convergenceDelta value")
                .withShortName("delta").create();
        
        Option maxIterationsOpt = obuilder
                .withLongName("maxIterations")
                .withArgument(abuilder.withName("maxIterations").withMinimum(1).withMaximum(1).create())
                .withDescription("k-means maxIterations value")
                .withShortName("mi").create();
        
        Option distanceMeasureOpt = obuilder
                .withLongName("distanceMeasure")
                .withArgument(abuilder.withName("distanceMeasure").withMinimum(1).withMaximum(1).create())
                .withDescription("k-means distance measure class name")
                .withShortName("dm").create();

        Group group = gbuilder.withName("Options").withOption(minSupportOpt)
                .withOption(chunkSizeOpt)
                .withOption(minDFOpt)
                .withOption(maxDFSigmaOpt).withOption(maxDFPercentOpt)
                .withOption(powerOpt)
                .withOption(minLLROpt).withOption(numReduceTasksOpt)
                .withOption(maxNGramSizeOpt)
                .withOption(helpOpt).withOption(sequentialAccessVectorOpt)
                .withOption(namedVectorOpt).withOption(logNormalizeOpt)
                .withOption(opTypeOpt).withOption(inputDirOpt)
                .withOption(kValueOpt)
                .withOption(convergenceDeltaOpt)
                .withOption(maxIterationsOpt)
                .withOption(distanceMeasureOpt)
                .create();
        try {
            Parser parser = new Parser();
            parser.setGroup(group);
            parser.setHelpOption(helpOpt);
            CommandLine cmdLine = parser.parse(args);

            if (cmdLine.hasOption(helpOpt)) {
                CommandLineUtil.printHelp(group);
                return false;
            }
            
            chunkSize = 100;
            if (cmdLine.hasOption(chunkSizeOpt)) {
                chunkSize = Integer.parseInt((String) cmdLine
                        .getValue(chunkSizeOpt));
            }
            log.info("chunkSize value: {}", chunkSize);
            
            minSupport = 2;
            if (cmdLine.hasOption(minSupportOpt)) {
                String minSupportString = (String) cmdLine
                        .getValue(minSupportOpt);
                minSupport = Integer.parseInt(minSupportString);
            }
            log.info("minSupport value: {}", minSupport);

            maxNGramSize = 1;
            if (cmdLine.hasOption(maxNGramSizeOpt)) {
                try {
                    maxNGramSize = Integer.parseInt(cmdLine.getValue(
                            maxNGramSizeOpt).toString());
                } catch (NumberFormatException ex) {
                    log.warn("Could not parse ngram size option");
                }
            }
            log.info("Maximum n-gram size is: {}", maxNGramSize);

            minLLRValue = LLRReducer.DEFAULT_MIN_LLR;
            if (cmdLine.hasOption(minLLROpt)) {
                minLLRValue = Float.parseFloat(cmdLine.getValue(minLLROpt)
                        .toString());
            }
            log.info("Minimum LLR value: {}", minLLRValue);

            reduceTasks = 1;
            if (cmdLine.hasOption(numReduceTasksOpt)) {
                reduceTasks = Integer.parseInt(cmdLine.getValue(
                        numReduceTasksOpt).toString());
            }
            log.info("Number of reduce tasks: {}", reduceTasks);
            
            minDf = 1;
            if (cmdLine.hasOption(minDFOpt)) {
              minDf = Integer.parseInt(cmdLine.getValue(minDFOpt).toString());
            }
            log.info("minDf Value: {}", minDf);
            maxDFPercent = 99;
            if (cmdLine.hasOption(maxDFPercentOpt)) {
              maxDFPercent = Integer.parseInt(cmdLine.getValue(maxDFPercentOpt).toString());
            }
            log.info("maxDFPercent Value: {}", maxDFPercent);
            maxDFSigma = -1.0;
            if (cmdLine.hasOption(maxDFSigmaOpt)) {
              maxDFSigma = Double.parseDouble(cmdLine.getValue(maxDFSigmaOpt).toString());
            }
            log.info("maxDFSigma Value: {}", maxDFSigma);

            norm = PartialVectorMerger.NO_NORMALIZING;
            if (cmdLine.hasOption(powerOpt)) {
              String power = cmdLine.getValue(powerOpt).toString();
              if ("INF".equals(power)) {
                norm = Float.POSITIVE_INFINITY;
              } else {
                norm = Float.parseFloat(power);
              }
            }
            log.info("norm Value: {}", norm);

            logNormalize = false;
            if (cmdLine.hasOption(logNormalizeOpt)) {
              logNormalize = true;
            }
            log.info("logNormalize Value: {}", logNormalize);
            
            sequentialAccessOutput = false;
            if (cmdLine.hasOption(sequentialAccessVectorOpt)) {
              sequentialAccessOutput = true;
            }
            log.info("sequentialAccessOutput Value: {}", sequentialAccessOutput);

            namedVectors = false;
            if (cmdLine.hasOption(namedVectorOpt)) {
              namedVectors = true;
            }
            log.info("namedVectors Value: {}", namedVectors);
            
            opType = "";
            if (cmdLine.hasOption(opTypeOpt)) {
                opType = cmdLine.getValue(opTypeOpt).toString();
            }
            log.info("opType Value: " + opType);
            
            inputDir = "";
            if (cmdLine.hasOption(inputDirOpt)) {
                inputDir = cmdLine.getValue(inputDirOpt).toString();
            }
            log.info("InputDir Value: {}", inputDir);
            
            kValue = 3;
            if (cmdLine.hasOption(kValueOpt)) {
                kValue = Integer.parseInt((String) cmdLine
                        .getValue(kValueOpt));
            }
            log.info("kmeans k value: {}", kValue);

            convergenceDelta = 0.01;
            if (cmdLine.hasOption(convergenceDeltaOpt)) {
                convergenceDelta = Double.parseDouble((String) cmdLine
                        .getValue(convergenceDeltaOpt));
            }
            log.info("kmeans convergenceDelta value: {}", convergenceDelta);
            
            maxIterations = 3;
            if (cmdLine.hasOption(maxIterationsOpt)) {
                maxIterations = Integer.parseInt((String) cmdLine
                        .getValue(maxIterationsOpt));
            }
            log.info("kmeans maxIterations value: {}", maxIterations);
            
            String measureClass = null;
            if (cmdLine.hasOption(distanceMeasureOpt)) {
                measureClass = cmdLine.getValue(distanceMeasureOpt).toString();
            }
            if (measureClass == null) {
              measureClass = SquaredEuclideanDistanceMeasure.class.getName();
            }
            measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
            log.info("kmeans measureClass value: {}", measureClass);
            
        } catch (OptionException e) {
            CommandLineUtil.printHelp(group);
            log.error("parse para error", e);
        }
        return true;
    }
    
    
    /**
     * read text file from hdfs and convert it to sequence file
     * input data format : video_id tag1 tag2 ...... tagN
     * output data format : key-Text, Value-StringTuple
     * @throws IOException 
     */
    @SuppressWarnings("deprecation")
    private void text2seq(String localTextFile) throws IOException {
        String stopWordsRegx = "(优酷)*(原创)*(拍客)*(自拍)*(视频)*(牛人)*(dv)*(null)*|(20|19)\\d\\d(年)*";
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        
        // local text file
        File file = new File(localTextFile);
        
        // output seq file
        Path originalDataDir = new Path(JOB_PATH, ORIGINAL_DATA_PATH);
        Path seqData = new Path(originalDataDir, "seqdata");
        // delete old seq data
        HadoopUtil.delete(conf, seqData);
        
        log.info("\n@\n@\n@\n");
        // prepare local file reader
        BufferedReader reader = null;
        reader = new BufferedReader(new FileReader(file));
        // prepare seq file writer, this will create a empty file
        SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
                seqData, Text.class, StringTuple.class);
        
        String tempString = null;
        log.info("wiriting data to seq file : " + seqData);
        while ((tempString = reader.readLine()) != null) {
            String[] data = tempString.split("\\|");
            if (data.length < MIN_TAG_COUNT) {
                continue;
            }
            String id = data[0];
            String valueString = "";
            StringTuple tags = new StringTuple();
            for (int i = 1; i < data.length - 1; i++) {
                valueString = data[i].toLowerCase();
                // TODO stop words filter
                if (valueString.matches(stopWordsRegx)) {
                    continue;
                }
                tags.add(valueString);
            }
            writer.append(new Text(id), tags);
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
        
        log.info("\n@\n@\n@\n");
        log.info("Calculating TF into : " + vectorDir + tfDirName);
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
        log.info("Calculating TF-vector done.");
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
        log.info("\n@\n@\n@\n");
        log.info("Calculating IDF into : " + dfDir);
        Pair<Long[], List<Path>> docFrequenciesFeatures = 
                TFIDFConverter.calculateDF(tfVectorDir, 
                        dfDir, 
                        conf, 
                        chunkSize);
        log.info("Calculating DF done.");
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
        
        log.info("\n@\n@\n@\n");
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
        log.info("Calculated maxDFThreshold is : " + maxDFThreshold);
        
        // prune the term frequency vectors
        log.info("Pruning tf-vector into : " + prunedTFVectorDir);
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
        log.info("Prune TF-vector done.");
    }
    
    private void calculateTfIdf(Pair<Long[], List<Path>> docFrequenciesFeatures)
            throws IOException, InterruptedException, ClassNotFoundException {
        // vector data path
        Path vectorDir = new Path(JOB_PATH, VECTOR_PATH);
        // pruned tf-vector path
        String prunedTfVectorDirName = "tf-vectors-pruned";
        Path prunedTFVectorDir = new Path(vectorDir, prunedTfVectorDirName);
        // tf-idf weighted vector path
        // tfidfVectorDirName should be same as TFIDFConverter.DOCUMENT_VECTOR_OUTPUT_FOLDER which
        // is invisible for public
        String tfidfVectorDirName = "tfidf-vectors";
        Path tfidfVectorDir = new Path(vectorDir, tfidfVectorDirName);
        
        // delete old data
        HadoopUtil.delete(conf, tfidfVectorDir);
        
        // calculate tf-idf weight
        // this method will generate tfidf-vectors into "tfidf-vectors" directory automatically, 
        // so the output para just need vectors' home path "video_tags_kmean_job/vectors"
        log.info("\n@\n@\n@\n");
        log.info("Calculating tfidf-vector into : " + tfidfVectorDir);
        TFIDFConverter.processTfIdf(
                prunedTFVectorDir,
                vectorDir, 
                conf, 
                docFrequenciesFeatures, 
                minDf, 
                maxDFPercent, 
                norm, 
                logNormalize,
                sequentialAccessOutput, 
                namedVectors, 
                reduceTasks);
        log.info("Calculating tfidf-vector done");
    }
    
    private void doClusteringJob() throws IOException, InterruptedException, ClassNotFoundException {
        // vector data path
        Path vectorDir = new Path(JOB_PATH, VECTOR_PATH);
        // tfidf-vector path
        String tfidfVectorDirName = "tfidf-vectors";
        Path tfidfVectorDir = new Path(vectorDir, tfidfVectorDirName);
        
        // cluster data path
        Path clusterPath = new Path(JOB_PATH, CLUSTER_PATH);
        
        HadoopUtil.delete(conf, clusterPath);
        
        log.info("\n@\n@\n@\n");
        log.info("Clusters' path : " + clusterPath);
        Path initCluster = new Path(clusterPath, "random-seeds");
        log.info("Running random seed to get initial clusters : " + initCluster);
        // choose random init clusters
        initCluster = RandomSeedGenerator.buildRandom(conf, 
                tfidfVectorDir, 
                initCluster, 
                kValue, 
                measure);
        
        log.info("Running KMeans with k = {}", kValue);
        // run k-means job
        KMeansDriver.run(conf, 
                tfidfVectorDir, 
                initCluster, 
                clusterPath, 
                convergenceDelta,
                maxIterations, 
                true, 
                0.0, 
                false);
        log.info("KMeans job done.");
    }

    private void doClusteringJobWithCanopy() throws Exception {
        // vector data path
        Path vectorDir = new Path(JOB_PATH, VECTOR_PATH);
        // tfidf-vector path
        String tfidfVectorDirName = "tfidf-vectors";
        Path tfidfVectorDir = new Path(vectorDir, tfidfVectorDirName);
        
        // cluster data path
        Path clusterPath = new Path(JOB_PATH, CLUSTER_PATH);
        HadoopUtil.delete(conf, clusterPath);
        
        log.info("\n@\n@\n@\n");
        
        log.info("Start to calculate t value...");
        log.info("Reading tfidf-vectors...");
        FileSystem fs = tfidfVectorDir.getFileSystem(conf);
        @SuppressWarnings("deprecation")
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(tfidfVectorDir, "part-r-00000"), conf);
        Text key = new Text();
        VectorWritable value = new VectorWritable();
        List<Vector> tifidfVectors = new ArrayList<Vector>();
        while (reader.next(key, value)) {
            tifidfVectors.add(value.get());
        }
        reader.close();
        
        // choose 1/10 vectors from tfidf-vectors as random vectors
        log.info("System will choose " + (tifidfVectors.size() / 10) + " random vectors.");
        List<Vector> randomVectors = RandomPointsUtil.chooseRandomPoints(
                tifidfVectors, (tifidfVectors.size() / 10));
        double sum = 0;
        if (randomVectors.size() <= 1) {
            throw new Exception("Choose random vector failed! You need at least two vectors to calculate distance!");
        }
        log.info("Calculating t value...");
        for (int i = 1; i < randomVectors.size(); i++) {
            double d = measure.distance(randomVectors.get(i - 1), randomVectors.get(i));
            sum = sum + d;
            log.info("distance = " + d + " sum = " + sum);
        }
        double averageDistance = sum / (randomVectors.size() - 1);
        double t1 = averageDistance * 1.3;
        double t2 = t1 / 2;
        log.info("t1 = " + t1);
        log.info("t2 = " + t2);
        
        log.info("Clusters' path : " + clusterPath);
        Path initCluster = new Path(clusterPath, "canopy-centroids");
        log.info("Running canopy to get initial clusters : " + initCluster);
        CanopyDriver.run(tfidfVectorDir, 
                initCluster,
                measure, 
                t1, 
                t2, 
                false, 
                0.0, 
                false);
        
        KMeansDriver.run(conf, 
                tfidfVectorDir, 
                new Path(initCluster, "clusters-0-final"), 
                clusterPath, 
                convergenceDelta,
                maxIterations, 
                true, 
                0.0, 
                false);
        
//        FuzzyKMeansDriver.run(conf, 
//                tfidfVectorDir, 
//                new Path(initCluster, "clusters-0-final"), 
//                clusterPath, 
//                convergenceDelta, 
//                maxIterations,
//                3, 
//                true, 
//                true, 
//                0.0, 
//                false);
        
        log.info("KMeans job done.");
    }
    
    private void dumpResult() throws Exception {
        log.info("\n@\n@\n@\n");
        Path clusterOutputPath = new Path(JOB_PATH, CLUSTER_PATH); // JOB_PATH + "/" + CLUSTER_PATH
        String finalClusterPath = null;
//        FileSystem fs = FileSystem.get(conf);
//        Path[] clusterPaths = FileUtil.stat2Paths(fs.listStatus(clusterOutputPath));
//        for (int i = 0; i < clusterPaths.length; i++) {
//            String path = clusterPaths[i].toString();
//            if (path.contains("final")) {
//                finalClusterPath = path;
//                break;
//            }
//        }
        
        // TODO to be tested.
        FileSystem fileSystem = clusterOutputPath.getFileSystem(conf);
        FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());
        finalClusterPath = clusterFiles[0].getPath().toString();
        
        if (finalClusterPath == null) {
            throw new Exception("Final cluster is not found !");
        }
        
        log.info("found final cluster {}", finalClusterPath);
        // use clusterdump to dump clusters
        String[] clusterDumpPara = {"-i", finalClusterPath,
                "-o", "video_tags_clusters_dump",
                "-d", "video_tags_kmean_job/vectors/dictionary.file-0",
                "-dt", "sequencefile",
                "-p", "video_tags_kmean_job/clusters/clusteredPoints",
                "-n", "50"};
        log.info("dumping clusters. para: " + Arrays.asList(clusterDumpPara).toString() );
        new MyClusterDumper().run(clusterDumpPara);
        
        // use seqdumper to dump videos and clusters
        String[] seqDumpPara = {"-i", "video_tags_kmean_job/clusters/clusteredPoints",
                "-o", "cluster_points_dump"};
        new SequenceFileDumper().run(seqDumpPara);
    }
    
    public static void main(String args[]) throws Exception {
        ToolRunner.run(new VideoTagsKMeansClustering(), args);
    }

    /**
     * hadoop jar mahout9all-Option.jar -ot 1 -i ~/data.txt -nv -seq -s 5 -md 3 -x 50
     * hadoop jar mahout9all-Option.jar -ot 2 -k 4 -mi 10 -delta 0.1 -dm org.apache.mahout.common.distance.CosineDistanceMeasure
     * mahout clusterdump -i video_tags_kmean_job/clusters/clusters-x-final -o ~/video_tags_clusters_dump -p video_tags_kmean_job/clusters/clusteredPoints -d video_tags_kmean_job/vectors/dictionary.file-0 -dt sequencefile -n 50
     * hadoop jar mahout9all-Option.jar -i ~/data.txt -nv -seq -s 5 -md 3 -x 50 -k 4 -mi 10 -delta 0.1 -dm org.apache.mahout.common.distance.CosineDistanceMeasure
     */
    public int run(String[] args) throws Exception {
        if (!init(args)) {
            log.error("Init failed !");
            return -1;
        }
        if (OP_TYPE_ONLY_VECTORIZE.equals(opType)) { // only prepare vectors data
            log.info("only prepare vectors data");
            text2seq(inputDir);
            calculateTF();
            Pair<Long[], List<Path>> docFrequenciesFeatures = calculateDF();
            pruneVectors(docFrequenciesFeatures);
            calculateTfIdf(docFrequenciesFeatures);
        } else if (OP_TYPE_ONLY_CLUSTERING.equals(opType)) { // only run kmeans job
            log.info("only run kmeans job");
            doClusteringJob();
//            doClusteringJobWithCanopy();
        } else if (OP_TYPE_ONLY_DUMP.equals(opType)) { // only dump result
            dumpResult();
        } else { // do all things
            log.info("do all things");
            text2seq(inputDir);
            calculateTF();
            Pair<Long[], List<Path>> docFrequenciesFeatures = calculateDF();
            pruneVectors(docFrequenciesFeatures);
            calculateTfIdf(docFrequenciesFeatures);
            doClusteringJob();
//            doClusteringJobWithCanopy();
            dumpResult();
        }
        return 0;
    }
    
}
