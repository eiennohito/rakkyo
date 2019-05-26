package ws.kotonoha.spark.tensorflow

import java.io.{Closeable, DataOutput, DataOutputStream}

import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.hadoop.mapred.{FileOutputFormat, JobConf, RecordWriter, Reporter}
import org.apache.hadoop.util.{Progressable, ReflectionUtils}
import ws.kotonoha.tensorflow.hadoop.util.TFRecordWriter

/**
  * @author eiennohito
  * @since 2017/01/20
  */
class TFRecordOutFormat extends FileOutputFormat[BytesWritable, NullWritable] {

  override def getRecordWriter(ignored: FileSystem, job: JobConf, name: String, progress: Progressable) = {

    val theOutput = if (FileOutputFormat.getCompressOutput(job)) {
      val codecClass = FileOutputFormat.getOutputCompressorClass(job, classOf[GzipCodec])
      val codec = ReflectionUtils.newInstance(codecClass, job)
      val path = FileOutputFormat.getTaskOutputPath(job, name + codec.getDefaultExtension)
      val fs = path.getFileSystem(job)
      val out = fs.create(path, progress)

      new DataOutputStream(
        codec.createOutputStream(out)
      )
    } else {
      val path = FileOutputFormat.getTaskOutputPath(job, name)
      val fs = path.getFileSystem(job)
      fs.create(path, progress)
    }

    new TFRecWriter(theOutput)
  }
}

class TFRecWriter(out: DataOutput with Closeable) extends RecordWriter[BytesWritable, NullWritable] {
  private val wrapped = new TFRecordWriter(out)

  override def write(key: BytesWritable, value: NullWritable) = {
    wrapped.write(key.getBytes, 0, key.getLength)
  }

  override def close(reporter: Reporter) = {
    out.close()
  }
}
