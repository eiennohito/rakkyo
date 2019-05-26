package ws.kotonoha.spark.tensorflow

import com.google.protobuf.{CodedInputStream, CodedOutputStream}
import com.trueaccord.scalapb.GeneratedMessage
import org.apache.hadoop.io.compress.CompressionCodec
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.tensorflow.example.example.{Example, SequenceExample}

/**
  * @author eiennohito
  * @since 2017/01/17
  */
object TFRecords {
  def readFromFile(sc: SparkContext, path: String): RDD[Example] = {
    sc.hadoopFile(
      path,
      classOf[TFRecordInFormat],
      classOf[BytesWritable],
      classOf[NullWritable]
    ).mapPartitions({ iter =>
      iter.map { case (bytes, _) =>
        val cos = CodedInputStream.newInstance(bytes.getBytes, 0, bytes.getLength)
        Example.parseFrom(cos)
      }
    }, preservesPartitioning = true)
  }

  def writeToFile(rdd: RDD[Example], path: String, codec: Option[Class[_ <: CompressionCodec]] = None): Unit = {
    TFImpl.writeToFile(rdd, path, codec)
  }
}

object TFImpl {
  def writeToFile[T <: GeneratedMessage](rdd: RDD[T], path: String, codec: Option[Class[_ <: CompressionCodec]] = None): Unit = {
    val res = rdd.mapPartitions({ iter =>
      new Iterator[(BytesWritable, NullWritable)] {
        private var buffer = new Array[Byte](64 * 1024) //64k buffer
        private var writable = new BytesWritable(buffer)
        override def hasNext: Boolean = iter.hasNext

        private def write(ex: T): Unit = {
          val size = ex.serializedSize
          if (buffer.length < size) {
            buffer = new Array[Byte](size * 11 / 10)
            writable = new BytesWritable(buffer)
          }
          val str = CodedOutputStream.newInstance(buffer)
          ex.writeTo(str)
          writable.setSize(size)
        }

        override def next(): (BytesWritable, NullWritable) = {
          write(iter.next())
          (writable, NullWritable.get())
        }
      }
    }, preservesPartitioning = true)

    res.saveAsHadoopFile(
      path = path,
      keyClass = classOf[BytesWritable],
      valueClass = classOf[NullWritable],
      outputFormatClass = classOf[TFRecordOutFormat],
      codec = codec
    )
  }
}

object TFSeqs {
  def writeToFile(rdd: RDD[SequenceExample], path: String, codec: Option[Class[_ <: CompressionCodec]] = None): Unit = {
    TFImpl.writeToFile(rdd, path, codec)
  }
}
