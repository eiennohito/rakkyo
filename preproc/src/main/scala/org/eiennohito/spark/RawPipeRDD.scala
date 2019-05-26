package org.eiennohito.spark

import java.lang.ProcessBuilder.Redirect
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicReference

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.hadoop.io.{BinaryComparable, NullWritable}
import org.apache.hadoop.mapred._
import org.apache.hadoop.util.{Progressable, ReflectionUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Partition, TaskContext}

import scala.collection.JavaConverters._

class RawPipeRDD(prev: RDD[(NullWritable, ByteBuffer)], command: Seq[String]) extends RDD[(NullWritable, ByteBuffer)](prev) {

  override def compute(split: Partition, context: TaskContext): Iterator[(NullWritable, ByteBuffer)] = {
    val p = new ProcessBuilder(command.asJava)

    p.redirectInput(Redirect.PIPE)
    p.redirectOutput(Redirect.PIPE)
    p.redirectError(Redirect.INHERIT)


    val proc = p.start()

    val childThrowable = new AtomicReference[Throwable](null)

    new Thread(s"stdin writer for $command") {
      override def run(): Unit = {
        val iter = prev.iterator(split, context)
        val os = proc.getOutputStream

        try {
          while (iter.hasNext) {
            val txt = iter.next()._2
            os.write(txt.array(), txt.position(), txt.remaining())
            txt.clear()
          }
        } catch {
          case e: Throwable => childThrowable.set(e)
        } finally {
          os.close()
        }

      }
    }.start()

    new Iterator[(NullWritable, ByteBuffer)] {
      private val input = proc.getInputStream
      private val buffer = ByteBuffer.allocate(16 * 1024)
      private var readBytes = 0

      override def hasNext: Boolean = {
        val child = childThrowable.get()
        if (child != null) {
          proc.destroy()
          throw new RuntimeException(s"child writer failed when processing $command", child)
        }

        if (readBytes > 0) {
          return true
        }

        while ({
          readBytes = input.read(buffer.array())
          readBytes == 0
        }) {}

        if (readBytes > 0) {
          return true
        }

        // read < 0; eof

        val exitCode = proc.waitFor()
        if (exitCode != 0) {
          throw new RuntimeException(s"process $command exited with code $exitCode")
        }

        false
      }

      override def next(): (NullWritable, ByteBuffer) = {
        buffer.limit(readBytes)
        readBytes = 0
        NullWritable.get() -> buffer
      }
    }
  }

  override protected def getPartitions: Array[Partition] = prev.partitions
}


class RawBytesOutputFormat extends FileOutputFormat[NullWritable, ByteBuffer] {
  override def getRecordWriter(ignored: FileSystem, conf: JobConf, name: String, progress: Progressable): RecordWriter[NullWritable, ByteBuffer] = {
    new RecordWriter[NullWritable, ByteBuffer] {
      private val outStream = {
        val ext = conf.get("rbof.extension", "")
        if (FileOutputFormat.getCompressOutput(conf)) {
          val codecClass = FileOutputFormat.getOutputCompressorClass(conf, classOf[GzipCodec])
          val codec = ReflectionUtils.newInstance(codecClass, conf)
          val path = if (ext.endsWith(codec.getDefaultExtension)) {
            FileOutputFormat.getTaskOutputPath(conf, name + ext)
          } else {
            FileOutputFormat.getTaskOutputPath(conf, name + ext + codec.getDefaultExtension)
          }
          val fs = path.getFileSystem(conf)
          val stream = fs.create(path, true, 16 * 1024)
          codec.createOutputStream(stream)
        } else {
          val file = FileOutputFormat.getTaskOutputPath(conf, name + ext)
          val fs = file.getFileSystem(conf)
          fs.create(file, true, 16 * 1024)
        }
      }

      override def write(key: NullWritable, value: ByteBuffer): Unit = {
        outStream.write(value.array(), value.position(), value.limit())
        progress.progress()
      }

      override def close(reporter: Reporter): Unit = {
        outStream.close()
      }
    }
  }

}