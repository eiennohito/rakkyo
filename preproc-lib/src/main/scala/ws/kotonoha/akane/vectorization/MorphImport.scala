package ws.kotonoha.akane.vectorization

import java.nio.file.Path

import scala.collection.mutable

class MorphTable(dics: IndexedSeq[TagDictionary]) {
  private[this] val table = new mutable.HashMap[(Int, Int), Set[String]]()

  def add(char: Int, tag: Int, value: String): Set[String] = {
    val key = (char, tag)
    val current = table.getOrElse(key, Set.empty)
    val updated = current + value
    table.update(key, updated)
    updated
  }

  def result(size: Int): Array[Int] = {
    val result = new Array[Int]((size + 2) * dics.size)
    var charIdx = 0
    while (charIdx < size) {
      var tagIdx = 0
      while (tagIdx < dics.size) {
        val key = (charIdx, tagIdx)
        val idx = (charIdx + 1) * dics.size + tagIdx

        table.get(key) match {
          case None => //noop
          case Some(s) =>
            if (s.size == 1) {
              val el = s.head
              val id = dics(tagIdx)(el)
              result(idx) = id
            }
        }

        tagIdx += 1
      }
      charIdx += 1
    }

    result
  }
}

class CharDictionary(
  encoding: Map[Int, Int],
  unk: Int
) {
  def apply(codept: Int): Int = encoding.getOrElse(codept, unk)
}

object CharDictionary {
  val nul = 0
  val unk = 1
  val bos = 2
  val eos = 3

  def load(p: Path): CharDictionary = {
    val pairs = DictionaryVectorizer.readPairsFromTsv(p, Int.MaxValue, fixed = true)

    val mapping = pairs.view.zipWithIndex.map {
      case ((c, _), i) => c.codePointAt(0) -> (i + 4)
    }.toMap

    new CharDictionary(mapping, unk)
  }

  val strings = IndexedSeq(
    "NUL",
    "UNK",
    "BOS",
    "EOS"
  )
}

class TagDictionary(
  tags: Map[String, Int]
) {
  def apply(tag: String): Int = tags.getOrElse(tag, 0)
}

object TagDictionary {
  def load(p: Path): TagDictionary = {
    val pairs = DictionaryVectorizer.readPairsFromTsv(p, Int.MaxValue, fixed = true)

    val mapping = pairs.view.zipWithIndex.map {
      case ((c, _), i) => c -> (i + 1)
    }.toMap

    new TagDictionary(mapping)
  }
}