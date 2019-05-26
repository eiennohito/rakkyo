package org.eiennohito.spark

import java.io.InputStream
import java.nio.ByteBuffer

import org.apache.hadoop.fs.{FileSystem, Path, Seekable}
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.hadoop.mapred._

class RawBytesInputFormat extends FileInputFormat[NullWritable, ByteBuffer] with JobConfigurable {

  var factory: CompressionCodecFactory = _

  override def configure(job: JobConf): Unit = {
    factory = new CompressionCodecFactory(job)
  }

  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter): RecordReader[NullWritable, ByteBuffer] = {
    val spl = split.asInstanceOf[FileSplit]
    val data = spl.getPath

    val fs = data.getFileSystem(job)
    val istream = fs.open(data, 4 * 1024 * 1024)

    val input = factory.getCodec(data) match {
      case null =>
        istream
      case codec =>
       codec.createInputStream(istream)
    }

    new ByteBufferReader(input, spl.getLength)
  }

  override def isSplitable(fs: FileSystem, filename: Path): Boolean = false
}

class ByteBufferReader(in: InputStream with Seekable, size: Long) extends RecordReader[NullWritable, ByteBuffer] {
  override def next(key: NullWritable, value: ByteBuffer): Boolean = {
    val nread = in.read(value.array(), value.position(), value.remaining())
    if (nread > 0) {
      value.limit(value.position() + nread)
    }

    nread >= 0
  }

  override def createKey(): NullWritable = NullWritable.get()
  override def createValue(): ByteBuffer = ByteBuffer.allocate(16 * 1024)

  override def getPos: Long = in.getPos
  override def close(): Unit = in.close()
  override def getProgress: Float = getPos.toFloat / size
}
