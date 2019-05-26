package ws.kotonoha.akane.vectorization

import java.io.{BufferedReader, InputStream, InputStreamReader, StringReader}
import java.nio.file.Path
import java.util.zip.GZIPInputStream

import org.apache.commons.io.IOUtils
import org.eiennohito.spark.{SeqHashing, VectorizerKeyer, XJppLatticeParser}
import org.tensorflow.example.example.Example
import ws.kotonoha.akane.analyzers.jumanpp.wire.LatticeNode
import ws.kotonoha.akane.io.{ByteBufferInputStream, Charsets, FrameAdapter}
import ws.kotonoha.akane.unicode.UnicodeUtil
import ws.kotonoha.spark.tensorflow.TFExample

import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.hashing.MurmurHash3

case class SentenceToken(
  tokens: Array[String]
) {
  override val hashCode: Int = MurmurHash3.seqHash(tokens)

  override def equals(that: Any): Boolean = {
    that match {
      case SentenceToken(otk) =>
        if (tokens.length != otk.length) {
          false
        } else {
          var i = 0
          val l = otk.length
          while (i < l) {
            val a = tokens(i)
            val b = otk(i)
            if (a != b) {
              return false
            }
            i += 1
          }
          true
        }
      case _ => false
    }
  }

  def mkString(sep: String): String = tokens.mkString(sep)
}

object SequenceToken {
  def apply(vals: String*) = SentenceToken(vals.toArray)

  def fromString(str: String, sep: String): SentenceToken = {
    val tokens = str.split(sep)
    SentenceToken(tokens)
  }
}

trait TokenProjection[T] {
  def project(o: T): SentenceToken
}

case class SentenceDifficultyEstimator(
  ranking: java.util.HashMap[SentenceToken, Int],
  default: Int
) {
  def rank[T](token: T)(implicit proj: TokenProjection[T]): Int = {
    val t = proj.project(token)
    ranking.getOrDefault(t, default)
  }

  override def toString: String = {
    s"SDE(size=${ranking.size} default=$default)"
  }
}

object SentenceDifficultyEstimator {

  import ws.kotonoha.akane.resources.FSPaths._

  def loadFromFile(file: Path, sep: String): SentenceDifficultyEstimator = {
    val mapbldr = new java.util.HashMap[SentenceToken, Int]
    var maxDiff = 0

    for (is <- file.inputStream) {
      val rdr = new BufferedReader(new InputStreamReader(is))

      var str: String = null
      while ( {
        str = rdr.readLine(); str != null
      }) {
        val tokens = str.split('\t')
        val raw = tokens(0)
        val diff = tokens(1).toInt
        mapbldr.put(SequenceToken.fromString(raw, sep), diff)
        maxDiff = maxDiff max diff
      }
    }
    new SentenceDifficultyEstimator(mapbldr, maxDiff + 1)
  }
}

object JumanppImplicits {

  implicit object JumanppTokenProjector extends TokenProjection[LatticeNode] {
    override def project(o: LatticeNode): SentenceToken = {
      val sP = o.stringPos.get
      SequenceToken(
        o.surface,
        o.reading,
        o.dicform,
        sP.pos,
        sP.subpos,
        sP.conjType,
        sP.conjForm
      )
    }
  }

}

class XMorphConverter(chdic: CharDictionary, tags: IndexedSeq[TagDictionary], maxLength: Int,
  unkCharsProb: Float = 0, zen2HanProb: Float = 0, ranker: SentenceDifficultyEstimator = null, topOne: Boolean = false, bert: Boolean = true) {

  val parser = new XJppLatticeParser()
  val keyer = new VectorizerKeyer(0, maxLength)

  def isFullWidth(x: String): Boolean = {
    UnicodeUtil.stream(x).forall(x => x >= (0xfee0 + 0x21) && x <= (0xfee0 + 0x7f))
  }

  def zen2Han(str: String): String = {
    val sb = new java.lang.StringBuilder(str.length)
    val iter = UnicodeUtil.stream(str)
    while (iter.hasNext) {
      val cp = iter.next()

      val diff = cp - 0xfee0
      if (diff >= 0x21 && diff <= 0xf7) {
        sb.appendCodePoint(diff)
      } else {
        sb.appendCodePoint(cp)
      }
    }

    sb.toString
  }

  def handle(raw: String): List[((Int, Int), Example)] = {
    try {
      val rdr = new BufferedReader(new StringReader(raw))
      handle0(rdr)
    } catch {
      case e: Exception =>
        e.printStackTrace()
        Nil
    }
  }

  def addBert(buffer: ArrayBuffer[Long], num: Int): Unit = {
    val option: Long = Random.nextFloat() match {
      case x if x < 0.10 => 1
      case x if x < 0.20 => 2
      case _ => 3
    }
    buffer.appendAll(Seq.fill(num)(option))
  }

  def handle0(rdr: BufferedReader): List[((Int, Int), Example)] = {
    val item = parser.parse(rdr)
    val table = new MorphTable(tags)

    val rawNodeIter = item.nodes.iterator
    val nodeIter = if (topOne) {
      rawNodeIter.filter(_.ranks.contains(1))
    } else rawNodeIter

    var prevNodeId = 0
    var sentCodeptSize = 0
    val chars = new ArrayBuffer[Long]()
    val unks = new ArrayBuffer[Long]()
    val bertBuffer = new ArrayBuffer[Long]()

    chars += CharDictionary.bos
    unks += 0
    bertBuffer += 0

    while (nodeIter.hasNext) {
      val n = nodeIter.next()

      val nodeCptSize = n.endIndex - n.startIndex

      if (n.prevNodes.contains(prevNodeId) && n.ranks.contains(1)) {
        prevNodeId = n.nodeId
        sentCodeptSize += nodeCptSize

        var surface = n.surface
        if (Random.nextFloat() < zen2HanProb && isFullWidth(surface)) {
          surface = zen2Han(surface)
        }

        // POS 1 == 特殊
        if (nodeCptSize == 1 && n.pos.pos == 1 && Random.nextFloat() < unkCharsProb) {
          chars += CharDictionary.unk
        } else {
          val charCodes = UnicodeUtil.stream(surface).map(cp => chdic(cp).toLong)
          chars ++= charCodes
        }

        // 7% chance to bert the whole token
        if (bert && Random.nextFloat() < 0.07) {
          addBert(bertBuffer, nodeCptSize)
        } else {
          for (_ <- 0 until nodeCptSize) {
            // 8% chance to bert a single codepoint
            if (bert && Random.nextFloat() < (0.08 * 0.93)) {
              addBert(bertBuffer, 1)
            } else {
              bertBuffer.append(0)
            }
          }
        }
      }

      val isUnk = n.features.exists(_.key == "未知語")

      var charIdx = n.startIndex
      while (charIdx < n.endIndex) {

        //segmentation
        val segTag = charIdx match {
          case x if x == n.startIndex => "B"
          case x if x == (n.endIndex - 1) => "E"
          case _ => "I"
        }
        table.add(charIdx, 0, segTag)
        val p = n.stringPos.get
        table.add(charIdx, 1, p.pos)
        table.add(charIdx, 2, p.subpos)
        table.add(charIdx, 3, p.conjType)
        table.add(charIdx, 4, p.conjForm)

        unks += (if (isUnk) 1L else 0L)

        charIdx += 1
      }
    }

    chars += CharDictionary.eos
    unks += 0
    bertBuffer += 0

    if (chars.size > maxLength) {
      return Nil
    }

    val ids = table.result(sentCodeptSize)
    assert(chars.size == (sentCodeptSize + 2))


    val hashCode = SeqHashing.hashSeq(chars)
    val key = if (ranker != null) {
      var rank = 0
      for (n <- item.nodes) {
        if (n.ranks.contains(1)) {
          import JumanppImplicits.JumanppTokenProjector
          val nodeRank = ranker.rank(n)
          rank = rank max nodeRank
        }
      }
      (rank, hashCode)
    } else {
      val len = Math.max(4, chars.size + Random.nextInt(7) - 3)
      keyer.keyFor(len, hashCode)
    }

    key -> TFExample.apply(
      "chars" -> TFExample.i64seq(chars),
      "tags" -> TFExample.i64seq(ids.map(_.toLong)),
      "unks" -> TFExample.i64seq(unks),
      "bert" -> TFExample.i64seq(bertBuffer)
    ) :: Nil
  }
}


object XMorphCorpusDiffStats {

  import ws.kotonoha.akane.resources.FSPaths._

  def openMaybeCompressed(p: Path): AutoClosableWrapper[InputStream] = {
    val opened = p.inputStream
    p.extension match {
      case "gz" => new AutoClosableWrapper(new GZIPInputStream(opened.obj))
      case _ => opened
    }
  }


  def main(args: Array[String]): Unit = {
    val dicdir = args(0).p
    val input = args(1).p
    val diffdic = args(2).p

    val chardic = CharDictionary.load(dicdir / "chars.bycode.dic")
    val tagDicNames = IndexedSeq(
      "seg.dic",
      "pos.dic",
      "subpos.dic",
      "ctype.dic",
      "cform.dic"
    )

    val tagDicts = tagDicNames.map(nm => TagDictionary.load(dicdir / nm))

    val tagStrs = tagDicNames.map { nm =>
      val vdic = nm.replace(".dic", ".vdic")
      val vdicPath = dicdir / vdic
      vdicPath.lines().toIndexedSeq
    }

    val charStrDic = (dicdir / "chars.bycode.vdic").lines().toIndexedSeq

    val estimator = SentenceDifficultyEstimator.loadFromFile(diffdic, "\u0001")
    val xmc = new XMorphConverter(chardic, tagDicts, 1000, 1, 1, estimator)

    val diffs = new Array[Int](estimator.default + 1)

    val parser = new XJppLatticeParser

    for {
      is <- openMaybeCompressed(input)
      frame <- FrameAdapter(is, "EOS\n".getBytes)
    } {
      val rdr = new BufferedReader(new InputStreamReader(new ByteBufferInputStream(frame), Charsets.utf8))
      val ((diff, _), data) :: _ = xmc.handle0(rdr)
      diffs(diff) += 1
    }

    for (idx <- diffs.indices) {
      println(s"$idx\t${diffs(idx)}")
    }
  }
}

object XMorphConverterDebug {

  import ws.kotonoha.akane.resources.FSPaths._

  def main(args: Array[String]): Unit = {
    val dicdir = args(0).p
    val input = args(1).p

    val chardic = CharDictionary.load(dicdir / "chars.bycode.dic")
    val tagDicNames = IndexedSeq(
      "seg.dic",
      "pos.dic",
      "subpos.dic",
      "ctype.dic",
      "cform.dic"
    )

    val tagDicts = tagDicNames.map(nm => TagDictionary.load(dicdir / nm))

    val tagStrs = tagDicNames.map { nm =>
      val vdic = nm.replace(".dic", ".vdic")
      val vdicPath = dicdir / vdic
      vdicPath.lines().toIndexedSeq
    }

    val charStrDic = (dicdir / "chars.bycode.vdic").lines().toIndexedSeq

    val xmc = new XMorphConverter(chardic, tagDicts, 1000, 1, 1)
    val stringData = IOUtils.toString(input.toUri)
    val (k, v) = xmc.handle(stringData).head
    val chars = v.features.get.feature("chars").getInt64List.value.toIndexedSeq
    val tags = v.features.get.feature("tags").getInt64List.value.toIndexedSeq

    val nchars = chars.size

    for (i <- 0 until nchars) {
      print(chars(i))
      for (j <- tagDicNames.indices) {
        val idx = i * tagDicNames.size + j
        print("\t")
        print(tags(idx))
      }
      println()
    }

    for (i <- 0 until nchars) {
      print(charStrDic(chars(i).toInt))
      for (j <- tagDicNames.indices) {
        val idx = i * tagDicNames.size + j
        print("\t")
        val tagId = tags(idx)
        val tagTbl = tagStrs(j)
        print(tagTbl(tagId.toInt))
      }
      println()
    }
  }
}