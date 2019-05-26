package org.eiennohito.spark

import java.nio.{ByteBuffer, CharBuffer}

import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.lib.input.{LineRecordReader, TextInputFormat}
import org.apache.hadoop.mapreduce.{InputSplit, TaskAttemptContext}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import ws.kotonoha.akane.io.Charsets

object EosDelimited {

  private [this] val delimiter: Array[Byte] = "EOS\n".getBytes(Charsets.utf8)

  private class KnpTableInputFormat extends TextInputFormat {
    override def createRecordReader(split: InputSplit, context: TaskAttemptContext): LineRecordReader = {
      new LineRecordReader(delimiter)
    }
  }

  def readFrom(sc: SparkContext, name: String): RDD[String] = {
    sc.newAPIHadoopFile(name,
      classOf[KnpTableInputFormat],
      classOf[LongWritable],
      classOf[Text]
    )
  }.mapPartitions({ iter =>
    new Iterator[String] {
      private var charBuf = CharBuffer.allocate(128 * 1024)
      private val decoder = Charsets.utf8.newDecoder()

      override def next(): String = {
        val nextItem = iter.next()
        val t = nextItem._2
        val bytes = t.getBytes
        val buf = ByteBuffer.wrap(bytes, 0, t.getLength)

        charBuf.clear()

        var res = decoder.decode(buf, charBuf, true)
        if (res.isOverflow || charBuf.remaining() < 4) {
          charBuf = CharBuffer.allocate(bytes.length + 4)
          res = decoder.decode(buf, charBuf, true)
        }
        if (res.isError) {
          res.throwException()
        }

        charBuf.put("EOS")
        charBuf.flip()
        charBuf.toString
      }

      override def hasNext: Boolean = iter.hasNext
    }
  }, preservesPartitioning = true)

}
