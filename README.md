# What is Rakkyo

This is code which was used for experiments for NAACL 2019 Paper:
*Shrinking Japanese Morphological Analyzers With Neural Networks and Semi-supervised Learning*.
Arseny Tolmachev, Daisuke Kawahara and Sadao Kurohashi. ([pdf](https://www.aclweb.org/anthology/N19-1281), [bibtex](https://aclweb.org/anthology/papers/N/N19/N19-1281.bib))

# Structure

This is a mixed Scala/Spark and Python/TensorFlow 1.x project.

**Warning**: code is of research quality and pretty unstructured.
Beware of dragons, dirty hacks and spaghetti.

You need to use Python 3.6+.
Dependencies are Tensofrlow 1.10+ (but not 2.0 series), pyhocon, matplotlib

For compiling preprocessing code you need to have JDK 1.8 installed with sbt 1.0+.
For running it you need to have Spark 2.3.1 installed (standalone mode without Hadoop is OK).

## Compiling preprocessing code

```$ sbt assembly```

`rakkyo/preproc/target/scala-2.11/preproc-assembly-0.1.0-SNAPSHOT.jar` will contain compilation results.

# How to preprocess data


You need to prepare the training data (TFExamples) using Spark,
then you can train Rakkyo models.


Rakkyo uses a lot of training data.
We were using Apache Spark (with Hadoop as storage) for data preprocessing.
It is possible to use Apache Spark without HDFS cluster if you can mount the data
at the same location on all computational nodes.

You will need Spark 2.3.1 to launch Spark Applications if you use the binary from Releases.
You can change Spark version in build.sbt and build preprocessing tools for other Spark versions.

## Making raw silver data with Juman++

First, we need to analyse some sentences using [Juman++](https://github.com/ku-nlp/jumanpp).
StreamLines2 class is a Spark application that launches external application,
one per partition (input file), which should take its input from stdin and write output to stdout.
StreamLines2 outputs a file for each input partition.


```
spark-submit.sh \
        --master $SPARK_URL \
        --class org.eiennohito.spark.StreamLines2 \
        local:/path/to/preproc-assembly-0.1.0-SNAPSHOT.jar \
        --input={raw sentences, can use globs or several paths here} \
        --output={files will be placed here} \
        --command=/path/to/jumanpp-launch.sh
```

Where `jumanpp-launch.sh` is a script like

```
#!/bin/bash
# set -x
set -e
set -o pipefail

# set niceness of self and children
renice -n 19 $$ >/dev/null 2>/dev/null || true

JUMANPP_BINARY=jumanpp
JUMANPP_MODEL=/path/to/jumanpp.model
JUMANPP_CONFIG=/path/to/jumanpp.conf


exec "$JUMANPP_BINARY" --model="$JUMANPP_MODEL" --config="$JUMANPP_CONFIG" -s1 --auto-nbest=2:8:15
```

## Making dictionaries

Next step creates dictionaries for converting characters and categorical strings
into integers for TFExamples.

```
spark-submit.sh \
    --master $SPARK_URL \
    --executor-memory 10g \
    --class org.eiennohito.spark.JumanppToTags \
    local:/path/to/preproc-assembly-0.1.0-SNAPSHOT.jar \
    --input=... \
    --output=... \
    --max-chars=40000
```

The last paramters specifies the maximum number of characters (treated as unicode codepoints) in character dictionary.
In our experiments, ~3B Japanese corpus had 18k unique characters in total.

## Making TFExamples

```
spark-submit.sh \
        --master $SPARK_URL \
        --driver-memory 60g \
        --executor-memory 40g \
        --class org.eiennohito.spark.JumanppToXMorph \
        --conf spark.driver.maxResultSize=40G \
        local:/path/to/preproc-assembly-0.1.0-SNAPSHOT.jar \
        --input=... \
        --dicts=/path/to/output_of_previous_step \
        --output=... \
        --stats=/statistics/on/sampling/will_be_put_here \
        --unk-symbol-prob=0.001 --zen2-han-prob=0.05 --max-length=150 \
        --diff-dict=/path/to/20p.01.20k.dic \
        --sample-ratio=0.16667 --boundary-ratio=0.02 --output-files=500
```

* Dicts are dictionary files outputted by the previous step
* Stats are always local (relative to Spark Driver) paths
* Input and Output can be on HDFS and are handled by Hadoop
* Diff Dict is morpheme difficulty dictionary (computed from morpheme unigram frequencies), you need to uncompress it to use http://lotus.kuee.kyoto-u.ac.jp/~arseny/rakkyo/20p.01.20k.dic.bz2
* `unk-symbol-prob` replaces some symbols (single codepoint with 特集-記号 Jumandic POS) with UNK character
* We also sample sentences proportional to their difficulty (defined as difficulty of the hardest morpheme in the sentence), with assigning lower weight to extreme (both low and high difficulty) sentences. Sample ratio controls the total final number of sentences and boundary ratio are how much sentences with extreme difficulties should be sampled.
* This program outputs .gz compressed TFRecords

# How to train models

You need to use configs.
See rakkyo-conf directory for examples.

Basic launch is:
```
python3 rakkyo.py config.conf shapshot_dir=/path/to/snapshots
```

# How to do inference

TODO