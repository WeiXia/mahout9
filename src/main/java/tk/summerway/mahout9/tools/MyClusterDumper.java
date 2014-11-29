/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package tk.summerway.mahout9.tools;

import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.clustering.cdbw.CDbwEvaluator;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.evaluation.ClusterEvaluator;
import org.apache.mahout.clustering.evaluation.RepresentativePointsDriver;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.utils.clustering.CSVClusterWriter;
import org.apache.mahout.utils.clustering.ClusterDumperWriter;
import org.apache.mahout.utils.clustering.ClusterWriter;
import org.apache.mahout.utils.clustering.GraphMLClusterWriter;
import org.apache.mahout.utils.clustering.JsonClusterWriter;
import org.apache.mahout.utils.vectors.VectorHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import com.google.common.io.Files;

public final class MyClusterDumper extends AbstractJob {

    public static final String SAMPLE_POINTS = "samplePoints";
    DistanceMeasure measure;

    public enum OUTPUT_FORMAT {
        TEXT, CSV, GRAPH_ML, JSON,
    }

    public static final String DICTIONARY_TYPE_OPTION = "dictionaryType";
    public static final String DICTIONARY_OPTION = "dictionary";
    public static final String POINTS_DIR_OPTION = "pointsDir";
    public static final String NUM_WORDS_OPTION = "numWords";
    public static final String SUBSTRING_OPTION = "substring";
    public static final String EVALUATE_CLUSTERS = "evaluate";

    public static final String OUTPUT_FORMAT_OPT = "outputFormat";

    private static final Logger log = LoggerFactory
            .getLogger(MyClusterDumper.class);
    private Path seqFileDir;
    private Path pointsDir;
    private long maxPointsPerCluster = Long.MAX_VALUE;
    private String termDictionary;
    private String dictionaryFormat;
    private int subString = Integer.MAX_VALUE;
    private int numTopFeatures = 10;
    private Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints;
    private OUTPUT_FORMAT outputFormat = OUTPUT_FORMAT.TEXT;
    private boolean runEvaluation;

    public MyClusterDumper(Path seqFileDir, Path pointsDir) {
        this.seqFileDir = seqFileDir;
        this.pointsDir = pointsDir;
        init();
    }

    public MyClusterDumper() {
        setConf(new Configuration());
    }

    public static void main(String[] args) throws Exception {
        new MyClusterDumper().run(args);
    }

    @Override
    public int run(String[] args) throws Exception {
        if (!buildParse(args)) {
            log.error("Parse parameters failed !");
            return -1;
        }
        init();
        printClusters(null);
        return 0;
    }

    private boolean buildParse(String[] args) {
        DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
        ArgumentBuilder abuilder = new ArgumentBuilder();
        GroupBuilder gbuilder = new GroupBuilder();

        Option inputDirOpt = DefaultOptionCreator.inputOption().create();
        Option outputDirOpt = DefaultOptionCreator.outputOption().create();

        Option outputFormatOpt = obuilder
                .withLongName(OUTPUT_FORMAT_OPT)
                .withArgument(abuilder.withName(OUTPUT_FORMAT_OPT).create())
                .withDescription(
                        "The optional output format for the results. Options: TEXT, CSV, JSON or GRAPH_ML. Default is TEXT")
                .withShortName("of").create();

        Option substringOpt = obuilder
                .withLongName(SUBSTRING_OPTION)
                .withArgument(abuilder.withName(SUBSTRING_OPTION).create())
                .withDescription(
                        "The number of chars of the asFormatString() to print")
                .withShortName("b").create();

        Option pointsDirOpt = obuilder
                .withLongName(POINTS_DIR_OPTION)
                .withArgument(abuilder.withName(POINTS_DIR_OPTION).create())
                .withDescription(
                        "The directory containing points sequence files mapping input vectors to their cluster. "
                                + "If specified, then the program will output the points associated with a cluster")
                .withShortName("p").create();

        Option samplePointsOpt = obuilder
                .withLongName(SAMPLE_POINTS)
                .withArgument(abuilder.withName(SAMPLE_POINTS).create())
                .withDescription(
                        "Specifies the maximum number of points to include _per_ cluster.  The default "
                                + "is to include all points")
                .withShortName("sp").create();

        Option dictionaryOpt = obuilder.withLongName(DICTIONARY_OPTION)
                .withArgument(abuilder.withName(DICTIONARY_OPTION).create())
                .withDescription("The dictionary file").withShortName("d")
                .create();

        Option dictionaryTypeOpt = obuilder
                .withLongName(DICTIONARY_TYPE_OPTION)
                .withArgument(
                        abuilder.withName(DICTIONARY_TYPE_OPTION).create())
                .withDescription(
                        "The dictionary file type (text|sequencefile), default is text")
                .withShortName("dt").create();

        Option numWordsOpt = obuilder.withLongName(NUM_WORDS_OPTION)
                .withArgument(abuilder.withName(NUM_WORDS_OPTION).create())
                .withDescription("The number of top terms to print")
                .withShortName("n").create();

        Option evaluateOpt = obuilder
                .withLongName(EVALUATE_CLUSTERS)
                .withArgument(abuilder.withName(EVALUATE_CLUSTERS).create())
                .withDescription(
                        "Run ClusterEvaluator and CDbwEvaluator over the input.  "
                                + "The output will be appended to the rest of the output at the end. Default is false.")
                .withShortName("e").create();

        Option distanceMeasureOpt = obuilder.withLongName("distanceMeasure")
                .withArgument(abuilder.withName("distanceMeasure").create())
                .withDescription("k-means distance measure class name")
                .withShortName("dm").create();

        Option helpOpt = obuilder.withLongName("help")
                .withDescription("Print out help").withShortName("h").create();

        Group group = gbuilder.withName("Options").withOption(inputDirOpt)
                .withOption(outputDirOpt).withOption(outputFormatOpt)
                .withOption(substringOpt).withOption(pointsDirOpt)
                .withOption(samplePointsOpt).withOption(dictionaryOpt)
                .withOption(dictionaryTypeOpt).withOption(numWordsOpt)
                .withOption(evaluateOpt).withOption(distanceMeasureOpt)
                .withOption(helpOpt).create();
        try {
            Parser parser = new Parser();
            parser.setGroup(group);
            parser.setHelpOption(helpOpt);
            CommandLine cmdLine = parser.parse(args);

            if (cmdLine.hasOption(helpOpt)) {
                CommandLineUtil.printHelp(group);
                return false;
            }

            seqFileDir = getInputPath();
            inputPath = getInputPath();
            inputFile = getInputFile();
            if (cmdLine.hasOption(inputDirOpt)) {
                seqFileDir = new Path(cmdLine.getValue(inputDirOpt).toString());
                inputPath = new Path(cmdLine.getValue(inputDirOpt).toString());
                inputFile = new File(cmdLine.getValue(inputDirOpt).toString());
            }
            log.info("seqFileDir value: {}", seqFileDir);
            log.info("inputPath value: {}", inputPath);
            log.info("inputFile value: {}", inputFile);

            outputPath = getOutputPath();
            outputFile = getOutputFile();
            if (cmdLine.hasOption(outputDirOpt)) {
                outputPath = new Path(cmdLine.getValue(outputDirOpt).toString());
                outputFile = new File(cmdLine.getValue(outputDirOpt).toString());
            }
            log.info("outputPath value: {}", outputPath);
            log.info("outputFile value: {}", outputFile);

            if (cmdLine.hasOption(pointsDirOpt)) {
                pointsDir = new Path(cmdLine.getValue(pointsDirOpt).toString());
            }
            log.info("pointsDir value: {}", pointsDir);

            if (cmdLine.hasOption(substringOpt)) {
                int sub = Integer.parseInt(cmdLine.getValue(substringOpt)
                        .toString());
                if (sub >= 0) {
                    subString = sub;
                }
            }
            log.info("subString value: {}", subString);

            termDictionary = cmdLine.getValue(dictionaryOpt).toString();
            dictionaryFormat = cmdLine.getValue(dictionaryTypeOpt).toString();
            log.info("termDictionary value: {}", termDictionary);
            log.info("dictionaryFormat value: {}", dictionaryFormat);

            if (cmdLine.hasOption(numWordsOpt)) {
                numTopFeatures = Integer.parseInt(cmdLine.getValue(numWordsOpt)
                        .toString());
            }
            log.info("numTopFeatures value: {}", numTopFeatures);

            outputFormat = OUTPUT_FORMAT.TEXT;
            if (cmdLine.hasOption(outputFormatOpt)) {
                outputFormat = OUTPUT_FORMAT.valueOf(cmdLine.getValue(
                        outputFormatOpt).toString());
            }
            log.info("outputFormat value: {}", outputFormat);

            if (cmdLine.hasOption(samplePointsOpt)) {
                maxPointsPerCluster = Long.parseLong(cmdLine.getValue(
                        samplePointsOpt).toString());
            } else {
                maxPointsPerCluster = Long.MAX_VALUE;
            }
            log.info("maxPointsPerCluster value: {}", maxPointsPerCluster);

            runEvaluation = cmdLine.hasOption(evaluateOpt);
            log.info("runEvaluation value: {}", runEvaluation);

            String distanceMeasureClass = null;
            if (cmdLine.hasOption(distanceMeasureOpt)) {
                distanceMeasureClass = cmdLine.getValue(distanceMeasureOpt)
                        .toString();
            }
            if (distanceMeasureClass != null) {
                measure = ClassUtils.instantiateAs(distanceMeasureClass,
                        DistanceMeasure.class);
            }
            log.info("distanceMeasureClass value: {}", distanceMeasureClass);

        } catch (OptionException e) {
            CommandLineUtil.printHelp(group);
            log.error("parse para error", e);
        }
        return true;
    }

    public void printClusters(String[] dictionary) throws Exception {
        Configuration conf = new Configuration();

        if (this.termDictionary != null) {
            if ("text".equals(dictionaryFormat)) {
                dictionary = VectorHelper.loadTermDictionary(new File(
                        this.termDictionary));
            } else if ("sequencefile".equals(dictionaryFormat)) {
                dictionary = VectorHelper.loadTermDictionary(conf,
                        this.termDictionary);
            } else {
                throw new IllegalArgumentException("Invalid dictionary format");
            }
        }

        Writer writer;
        boolean shouldClose;
        if (this.outputFile == null) {
            shouldClose = false;
            writer = new OutputStreamWriter(System.out, Charsets.UTF_8);
        } else {
            shouldClose = true;
            if (outputFile.getName().startsWith("s3n://")) {
                Path p = outputPath;
                FileSystem fs = FileSystem.get(p.toUri(), conf);
                writer = new OutputStreamWriter(fs.create(p), Charsets.UTF_8);
            } else {
                Files.createParentDirs(outputFile);
                writer = Files.newWriter(this.outputFile, Charsets.UTF_8);
            }
        }
        ClusterWriter clusterWriter = createClusterWriter(writer, dictionary);
        try {
            long numWritten = clusterWriter
                    .write(new SequenceFileDirValueIterable<ClusterWritable>(
                            new Path(seqFileDir, "part-*"), PathType.GLOB, conf));

            writer.flush();
            if (runEvaluation) {
                HadoopUtil.delete(conf, new Path("tmp/representative"));
                int numIters = 5;
                RepresentativePointsDriver.main(new String[] { "--input",
                        seqFileDir.toString(), "--output",
                        "tmp/representative", "--clusteredPoints",
                        pointsDir.toString(), "--distanceMeasure",
                        measure.getClass().getName(), "--maxIter",
                        String.valueOf(numIters) });
                conf.set(RepresentativePointsDriver.DISTANCE_MEASURE_KEY,
                        measure.getClass().getName());
                conf.set(RepresentativePointsDriver.STATE_IN_KEY,
                        "tmp/representative/representativePoints-" + numIters);
                ClusterEvaluator ce = new ClusterEvaluator(conf, seqFileDir);
                writer.append("\n");
                writer.append("Inter-Cluster Density: ")
                        .append(String.valueOf(ce.interClusterDensity()))
                        .append("\n");
                writer.append("Intra-Cluster Density: ")
                        .append(String.valueOf(ce.intraClusterDensity()))
                        .append("\n");
                CDbwEvaluator cdbw = new CDbwEvaluator(conf, seqFileDir);
                writer.append("CDbw Inter-Cluster Density: ")
                        .append(String.valueOf(cdbw.interClusterDensity()))
                        .append("\n");
                writer.append("CDbw Intra-Cluster Density: ")
                        .append(String.valueOf(cdbw.intraClusterDensity()))
                        .append("\n");
                writer.append("CDbw Separation: ")
                        .append(String.valueOf(cdbw.separation())).append("\n");
                writer.flush();
            }
            log.info("Wrote {} clusters", numWritten);
        } finally {
            if (shouldClose) {
                Closeables.close(clusterWriter, false);
            } else {
                if (clusterWriter instanceof GraphMLClusterWriter) {
                    clusterWriter.close();
                }
            }
        }
    }

    ClusterWriter createClusterWriter(Writer writer, String[] dictionary)
            throws IOException {
        ClusterWriter result;

        switch (outputFormat) {
        case TEXT:
            result = new ClusterDumperWriter(writer, clusterIdToPoints,
                    measure, numTopFeatures, dictionary, subString);
            break;
        case CSV:
            result = new CSVClusterWriter(writer, clusterIdToPoints, measure);
            break;
        case GRAPH_ML:
            result = new GraphMLClusterWriter(writer, clusterIdToPoints,
                    measure, numTopFeatures, dictionary, subString);
            break;
        case JSON:
            result = new JsonClusterWriter(writer, clusterIdToPoints, measure,
                    numTopFeatures, dictionary);
            break;
        default:
            throw new IllegalStateException("Unknown outputformat: "
                    + outputFormat);
        }
        return result;
    }

    /**
     * Convenience function to set the output format during testing.
     */
    public void setOutputFormat(OUTPUT_FORMAT of) {
        outputFormat = of;
    }

    private void init() {
        if (this.pointsDir != null) {
            Configuration conf = new Configuration();
            // read in the points
            clusterIdToPoints = readPoints(this.pointsDir, maxPointsPerCluster,
                    conf);
        } else {
            clusterIdToPoints = Collections.emptyMap();
        }
    }

    public int getSubString() {
        return subString;
    }

    public void setSubString(int subString) {
        this.subString = subString;
    }

    public Map<Integer, List<WeightedPropertyVectorWritable>> getClusterIdToPoints() {
        return clusterIdToPoints;
    }

    public String getTermDictionary() {
        return termDictionary;
    }

    public void setTermDictionary(String termDictionary, String dictionaryType) {
        this.termDictionary = termDictionary;
        this.dictionaryFormat = dictionaryType;
    }

    public void setNumTopFeatures(int num) {
        this.numTopFeatures = num;
    }

    public int getNumTopFeatures() {
        return this.numTopFeatures;
    }

    public long getMaxPointsPerCluster() {
        return maxPointsPerCluster;
    }

    public void setMaxPointsPerCluster(long maxPointsPerCluster) {
        this.maxPointsPerCluster = maxPointsPerCluster;
    }

    public static Map<Integer, List<WeightedPropertyVectorWritable>> readPoints(
            Path pointsPathDir, long maxPointsPerCluster, Configuration conf) {
        Map<Integer, List<WeightedPropertyVectorWritable>> result = Maps
                .newTreeMap();
        for (Pair<IntWritable, WeightedPropertyVectorWritable> record : new SequenceFileDirIterable<IntWritable, WeightedPropertyVectorWritable>(
                pointsPathDir, PathType.LIST, PathFilters.logsCRCFilter(), conf)) {
            // value is the cluster id as an int, key is the name/id of the
            // vector, but that doesn't matter because we only care about
            // printing it
            // String clusterId = value.toString();
            int keyValue = record.getFirst().get();
            List<WeightedPropertyVectorWritable> pointList = result
                    .get(keyValue);
            if (pointList == null) {
                pointList = Lists.newArrayList();
                result.put(keyValue, pointList);
            }
            if (pointList.size() < maxPointsPerCluster) {
                pointList.add(record.getSecond());
            }
        }
        return result;
    }
}
