package org.eiennohito.spark

import java.nio.ByteBuffer
import java.util.StringTokenizer

import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.{SparkConf, SparkContext}
import org.rogach.scallop.ScallopConf

import scala.collection.mutable.ArrayBuffer

object StreamLines {

  private class SLConfig(args: Seq[String]) extends ScallopConf(args) {
    val input = opt[String](required = true)
    val output = opt[String](required = true)
    val command = opt[String](required = true)
  }

  def main(args: Array[String]): Unit = {
    val a = new SLConfig(args)
    a.verify()

    val conf = new SparkConf().setAppName("CommandlineRunner")
    val sc = new SparkContext(conf)

    try {
      doWork(sc, a)
    } finally {
      sc.stop()
    }
  }

  def doWork(sc: SparkContext, cfg: SLConfig): Unit = {
    val inputs = sc.textFile(cfg.input())
    val lines = inputs.pipe(cfg.command())
    lines.saveAsTextFile(cfg.output(), classOf[GzipCodec])
  }
}


object StreamLines2 {

  private class SLConfig(args: Seq[String]) extends ScallopConf(args) {
    val input = opt[String](required = true)
    val output = opt[String](required = true)
    val command = opt[String](required = true)
  }

  def main(args: Array[String]): Unit = {
    val a = new SLConfig(args)
    a.verify()

    val conf = new SparkConf().setAppName("CommandlineRunner2")
    val sc = new SparkContext(conf)

    try {
      doWork(sc, a)
    } finally {
      sc.stop()
    }
  }

  def doWork(sc: SparkContext, cfg: SLConfig): Unit = {
    val inputs = sc.hadoopFile[NullWritable, ByteBuffer, RawBytesInputFormat](cfg.input())
    val lines = new RawPipeRDD(inputs, tokenize(cfg.command()))
    lines.saveAsHadoopFile(
      cfg.output(),
      classOf[NullWritable],
      classOf[ByteBuffer],
      classOf[RawBytesOutputFormat],
      classOf[GzipCodec]
    )
  }

  def tokenize(command: String): Seq[String] = {
    val buf = new ArrayBuffer[String]
    val tok = new StringTokenizer(command)
    while (tok.hasMoreElements) {
      buf += tok.nextToken()
    }
    buf
  }
}