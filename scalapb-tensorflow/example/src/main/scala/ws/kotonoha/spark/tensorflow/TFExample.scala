package ws.kotonoha.spark.tensorflow

import com.google.protobuf.ByteString
import org.tensorflow.example.example.Example
import org.tensorflow.example.feature.Feature.Kind
import org.tensorflow.example.feature._

/**
  * @author eiennohito
  * @since 2017/01/24
  */
object TFExample {
  def apply(data: (String, Feature)*): Example = Example(Some(Features(data.toMap)))
  def i64(data: Long*): Feature = i64seq(data)
  def f32(data: Float*): Feature = Feature(Kind.FloatList(FloatList(data.toArray)))
  def i64seq(data: Seq[Long]): Feature = Feature(Kind.Int64List(Int64List(data.toArray)))
  def byteSeq(data: Seq[ByteString]): Feature = {
    Feature(Kind.BytesList(
      BytesList(data)
    ))
  }
}
