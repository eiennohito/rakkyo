package ws.kotonoha.spark.tensorflow

import java.io.{DataInputStream, IOException}
import java.nio.{ByteBuffer, ByteOrder}

import org.apache.hadoop.fs.{FileSystem, Path, Seekable}
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.hadoop.mapred._
import ws.kotonoha.tensorflow.hadoop.io.TFRecordIOConf
import ws.kotonoha.tensorflow.util.Crc32C
/**
  * @author eiennohito
  * @since 2017/01/23
  */
class TFRecordInFormat extends FileInputFormat[BytesWritable, NullWritable] with JobConfigurable {

  private var codecs: CompressionCodecFactory = null

  override def configure(job: JobConf) = {
    codecs = new CompressionCodecFactory(job)
  }

  override def getRecordReader(split: InputSplit, job: JobConf, reporter: Reporter) = {
    reporter.setStatus(split.toString)
    TFRecordReader.make(split.asInstanceOf[FileSplit], job, reporter, codecs)
  }

  override def isSplitable(fs: FileSystem, filename: Path) = {
    false
  }
}

class TFRecordReader(
  input: DataInputStream,
  progress: Seekable,
  doCheckCrc: Boolean,
  length: Long
) extends RecordReader[BytesWritable, NullWritable] {
  private val wrapped = new ws.kotonoha.tensorflow.hadoop.util.TFRecordReader(input, doCheckCrc)

  private val tempBuffer = ByteBuffer.allocate(24)
  tempBuffer.order(ByteOrder.LITTLE_ENDIAN)

  private def doRead(buffer: Array[Byte], offset: Int, length: Int): Int = {
    var read = 0
    while (read < length) {
      val readNow = input.read(buffer, offset + read, length - read)
      if (readNow == -1) {
        return if (read == 0) -1 else read
      }
      read += readNow
    }
    read
  }

  override def next(key: BytesWritable, value: NullWritable): Boolean = {
    val hasRead = doRead(tempBuffer.array(), 0, 12)
    hasRead match {
      case -1 => return false //end of file, ok
      case 12 => //do nothing
      case _ =>
        throw new IOException(s"header is corrupted, there were no 12 bytes of header, was $hasRead bytes")
    }

    val msgSize = tempBuffer.getLong(0)

    if (doCheckCrc) {
      val sizeCrc = tempBuffer.getInt(8)
      val computedCrc = Crc32C.maskedCrc32c(tempBuffer.array(), 0, 8)
      if (computedCrc != sizeCrc) {
        throw new IOException(s"length header crc check failed for size=$msgSize, crc=$computedCrc, expected=$sizeCrc")
      }
    }

    val intSize = msgSize.toInt
    key.setSize(intSize)

    val actuallyRead = doRead(key.getBytes, 0, intSize)
    if (actuallyRead != intSize) {
      throw new IOException(s"corrupted file: should have read $intSize, read $actuallyRead")
    }

    if (doCheckCrc) {
      val haveRead = doRead(tempBuffer.array(), 12, 4)
      val dataCrc = tempBuffer.getInt(12)
      val actualCrc = Crc32C.maskedCrc32c(key.getBytes, 0, intSize)
      if (dataCrc != actualCrc) {
        throw new IOException(s"content corrupted: crc=$actualCrc, expected=$dataCrc, crcsize=$haveRead")
      }
    } else {
      input.skipBytes(4)
    }

    true
  }

  override def getProgress = (getPos.toDouble / length).toFloat
  override def getPos = progress.getPos
  override def close() = input.close()
  override def createKey() = new BytesWritable()
  override def createValue() = NullWritable.get()
}

object TFRecordReader {
  def make(split: FileSplit, job: JobConf, reporter: Reporter, codecs: CompressionCodecFactory): TFRecordReader = {
    val path = split.getPath
    val fs = path.getFileSystem(job)
    val (inp, prog) = codecs.getCodec(path) match {
      case null =>
        val f = fs.open(path)
        (f, f)
      case c =>
        val stream = fs.open(path)
        val decoded = c.createInputStream(stream)
        val ret = new DataInputStream(decoded)
        (ret, stream)
    }

    val checkCrc = TFRecordIOConf.getDoCrc32Check(job)

    new TFRecordReader(
      inp,
      prog,
      checkCrc,
      split.getLength
    )
  }
}
