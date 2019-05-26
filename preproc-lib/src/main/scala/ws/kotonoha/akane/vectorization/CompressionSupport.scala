package ws.kotonoha.akane.vectorization

import java.io.InputStream
import java.nio.file.Path
import java.util.zip.GZIPInputStream


/**
  * @author eiennohito
  * @since 2017/02/09
  */
object CompressionSupport {
  def wrapStream(is: InputStream, extension: String): InputStream = {
    extension match {
      case "gz" => new GZIPInputStream(is)
      case _ => is
    }
  }

  import ws.kotonoha.akane.resources.FSPaths._
  def wrapStream(is: InputStream, path: Path): InputStream = wrapStream(is, path.extension)
}
