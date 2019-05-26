package ws.kotonoha.akane.vectorization

import java.io.PrintWriter
import java.nio.file.Path

import org.apache.commons.lang3.StringUtils
import ws.kotonoha.akane.analyzers.juman.{JumanPos, JumanStylePos}
import ws.kotonoha.akane.analyzers.knp.LexemeApi
import ws.kotonoha.akane.checkers.JumanPosCheckers
import ws.kotonoha.akane.parser.JumanPosSet
import ws.kotonoha.akane.resources.FSPaths
import ws.kotonoha.akane.unicode.UnicodeUtil
import ws.kotonoha.akane.utils.XLong

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * @author eiennohito
  * @since 2017/01/31
  */

class DictionaryVectorizer(dic: Map[String, Int], subsampleFreq: Array[Float], feature: LexemeApi => String) {

  def shouldKeep(id: Long): Float = {
    if (id < DictionaryVectorizer.offset) {
      1.0f //have no statistics on strange words
    } else subsampleFreq(id.toInt - DictionaryVectorizer.offset)
  }

  def idFor(writ: String): Option[Int] = {
    idInternal(writ).map(_ + DictionaryVectorizer.offset)
  }

  private def idInternal(writ: String) = {
    dic.get(DictionaryVectorizer.norm(writ))
  }

  def vecOne(lex: LexemeApi): Seq[Long] = {
    val seq = Vector.newBuilder[Long]
    doVecOne(seq, lex)
    seq.result()
  }

  def id(word: LexemeApi): Int = {
    idInternal(feature(word)) match {
      case Some(x) => x + DictionaryVectorizer.offset
      case _ => DictionaryVectorizer.slowPath(word) + DictionaryVectorizer.hardcodedSize
    }
  }

  def vectorize(data: Iterator[LexemeApi], tokens: Boolean = true): Seq[Long] = {
    val bldr = Seq.newBuilder[Long]
    if (tokens) {
      bldr += DictionaryVectorizer.bos //BOS
    }

    while (data.hasNext) {
      val item = data.next()
      doVecOne(bldr, item)
    }

    if (tokens) {
      bldr += DictionaryVectorizer.eos //EOS
    }
    bldr.result()
  }

  private def doVecOne(bldr: mutable.Builder[Long, Seq[Long]], item: LexemeApi) = {
    bldr += id(item)
    if (item.pos.category != 0) {
      bldr += DictionaryVectorizer.conjs.idOf(item)
    }
  }
}

object DictionaryVectorizer {

  def readTsv2(file: Path, maxFull: Int, max: Int): IndexedSeq[(String, Long)] = {
    import FSPaths._

    val result = new ArrayBuffer[(String, Long)]()
    result.sizeHint(max)

    val lines = file.lines()

    try {
      while (lines.hasNext && result.size < max) {
        val line = lines.next()
        val pair = parseLine(line)

        if (result.size < maxFull || isViable(pair._1)) {
          result += pair
        }
      }
    } finally {
      lines.close()
    }

    result
  }

  private def isViable(str: String): Boolean = {
    if (str.indexOf('?') > 0) {
      return true
    }

    val slash = str.indexOf('/')
    if (slash <= 0) {
      return true
    }

    val left = str.substring(0, slash)
    val right = str.substring(slash + 1)

    left != right || UnicodeUtil.stream(left).exists(UnicodeUtil.isHiragana)
  }

  def readPairsFromTsv(file: Path, max: Int, fixed: Boolean = false): Vector[(String, Long)] = {
    import FSPaths._
    val lines = file.lines().take(max).toVector
    if (fixed) {
      fromLinesFixed(lines, max)
    } else {
      fromLines(lines, max)
    }
  }

  def fromLines(lines: Seq[String], max: Int): Vector[(String, Long)] = {
    lines.filter(_.length > 2).take(max - DictionaryVectorizer.offset).map { s =>
      parseLine(s)
    }.toVector
  }

  def fromLinesFixed(lines: Seq[String], max: Int): Vector[(String, Long)] = {
    lines.filter(_.length > 2).take(max - DictionaryVectorizer.offset).map { s =>
      parseLineFixed(s)
    }.toVector
  }

  private def parseLine(s: String) = {
    StringUtils.split(s, '\t') match {
      case Array(XLong(cnt), str) => (DictionaryVectorizer.norm(str), cnt)
      case Array(str, XLong(cnt)) => (DictionaryVectorizer.norm(str), cnt)
    }
  }

  private def parseLineFixed(s: String) = {
    StringUtils.split(s, '\t') match {
      case Array(str, XLong(cnt)) => (DictionaryVectorizer.norm(str), cnt)
      case _ => throw new Exception(s"Unparseable string: $s")
    }
  }

  def fromWordFreqPairs(data: TraversableOnce[(String, Long)], feature: LexemeApi => String): DictionaryVectorizer = {
    val buf = Array.newBuilder[Float]
    val mapBldr = Map.newBuilder[String, Int]
    var idx = 0
    var total = 0L

    val iter = data.toIterator
    while (iter.hasNext) {
      val (word, cnt) = iter.next()
      mapBldr += word -> idx
      idx += 1
      total += cnt
      buf += cnt
    }

    idx = 0
    val totalDbl = total.toDouble
    val probs = buf.result()
    while (idx < probs.length) {
      val cnt = probs(idx)
      probs(idx) = Math.min(1.0f, subsamplingProb(cnt / totalDbl).toFloat)
      idx += 1
    }

    new DictionaryVectorizer(mapBldr.result(), probs, feature)
  }


  def subsamplingProb(freq: Double) = (math.sqrt(freq / 1e-5) + 1) * (1e-5/freq)

  def norm(writ: String): String = {
    if (writ.indexOf('?') < 1) writ
    else StringUtils.split(writ, '?').sorted.mkString("?")
  }


  def conjOf(thepos: JumanPos): Int = conjs.idOfPos(thepos)

  val unkWordCheckers = {
    val ch = JumanPosCheckers.default
    Array(
      ch.posSubPos("名詞", "普通名詞"),
      ch.posSubPos("名詞", "数詞"),
      ch.pos("名詞"),
      ch.pos("動詞"),
      ch.pos("形容詞"),
      ch.pos("副詞"),
      ch.posSubPos("未定義語", "カタカナ"),
      ch.posSubPos("未定義語", "アルファベット")
    )
  }

  def restorator(data: IndexedSeq[String]): DictionaryRestorator = new DictionaryRestorator(checkers() ++ data)

  val hardcoded = IndexedSeq("NONE", "$$$", "BOS", "EOS", "UNK")
  val hardcodedSize: Int = hardcoded.size

  val question = 1
  val bos = 2
  val eos = 3
  val globalUnk = 4

  private val conjs = new ConjVectorizer(hardcodedSize + unkWordCheckers.length, ConjVectorizer.idxes)

  def checkers(): IndexedSeq[String] = {
    hardcoded ++ unkWordCheckers.map(s => s.toString.replace(' ', '_')) ++ ConjVectorizer.names
  }

  val offset: Int = hardcoded.length + unkWordCheckers.length + ConjVectorizer.names.length

  def slowPath(word: LexemeApi): Int = {
    var i = 0
    while (i < unkWordCheckers.length) {
      val ch = unkWordCheckers(i)
      if (ch.check(word)) {
        return i
      }
      i += 1
    }
    -1 //UNK
  }
}

class DictionaryRestorator(data: IndexedSeq[String]) {
  def restoreTo(context: Iterable[Long], osw: Appendable, separator: String = " "): Unit = {
    val iter = context.iterator
    var hn: Boolean = iter.hasNext

    while (hn) {
      osw.append(restoreOne(iter.next()))
      hn = iter.hasNext
      if (hn) {
        osw.append(separator)
      }
    }

  }

  def restoreOne(id: Long): String = {
    if (id >= 0 && id < data.length) {
      data(id.toInt)
    } else {
      "UNK"
    }
  }

  def restore(smt: TraversableOnce[Long]): String = {
    val bldr = new StringBuilder
    val iter = smt.toIterator

    while (iter.hasNext) {
      val x = data(iter.next().toInt)
      bldr.append(x).append(" ")
    }

    bldr.result()
  }

  def dump(path: Path): Unit = {
    import FSPaths._
    path.writeLines(data)
  }
}


object ConjVectorizer {
  val names: Array[String] = {
    JumanPosSet.default.conjugatons.flatMap(t => t.conjugations.filter(_.num != 0).map(_.name)).distinct
  }

  val idxes: Map[Int, Int] = {
    val nameToId = names.zipWithIndex.toMap
    val ids = JumanPosSet.default.conjugatons
      .flatMap(c => c.conjugations.filter(_.num != 0).map { f => idx(c.num, f.num) -> nameToId(f.name) })
    ids.toMap
  }

  def idx(cat: Int, conj: Int): Int = (cat << 16) | conj
}

class ConjVectorizer(start: Int, entries: Map[Int, Int]) {
  def idOfPos(pos: JumanStylePos): Int = {
    val idx = ConjVectorizer.idx(pos.category, pos.conjugation)
    entries(idx) + start
  }

  def idOf(l: LexemeApi): Int = {
    val pos = l.pos
    idOfPos(pos)
  }
}

object KnpFeatures {
  def featureExtractor(s: String): LexemeApi => String = {
    s match {
      case "norm" => (api) => api.valueOfFeature("正規化代表表記").getOrElse(api.canonicForm())
      case _ => (api) => api.canonicForm()
    }
  }
}
