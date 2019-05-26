package org.eiennohito.spark

import java.io.{BufferedReader, StringReader}
import java.nio.file.Path

import org.apache.spark.{SparkConf, SparkContext}
import org.rogach.scallop.ScallopConf
import ws.kotonoha.akane.parser.JumanPosSet
import ws.kotonoha.akane.unicode.UnicodeUtil
import ws.kotonoha.akane.vectorization.CharDictionary

object JumanppToTags {
  import ws.kotonoha.akane.resources.FSPaths._

  private class Jpp2TagsArgs(args: Seq[String]) extends ScallopConf(args) {
    val input = opt[Path](required = true)
    val output = opt[String](required = true)
    val maxChars = opt[Int](default = Some(20000))
  }

  val posser = JumanPosSet.default

  def doWork(sc: SparkContext, filenames: String, output: String, maxChars: Int): Unit = {
    val data = EosDelimited.readFrom(sc, filenames)

    val items = data.mapPartitions(iter => {
      val parser = new XJppLatticeParser()
      iter.flatMap { txt =>
        val data = txt.toString
        try {
          process(parser, data)
        } catch {
          case e: Exception =>
            e.printStackTrace()
            Nil
        }
      }
    })

    val result = items.map(_ -> 1L).reduceByKey(_ + _).take(maxChars * 2)

    val byType = result.groupBy(_._1._1).mapValues(_.view.map{ case ((_, k), c) => (k, c)}.toIndexedSeq)

    val asciiChars = (0x21 to 0x7f).map(c => Character.toString(c.toChar) -> 0L).toMap

    val chars = (asciiChars ++ byType("cpt")).toIndexedSeq.sortBy(-_._2)

    val base = output.p

    base.mkdirs()

    def codeptRepr(s: String): String = {
      val cpt = Character.codePointAt(s, 0)
      "%s:%x".format(new String(Character.toChars(cpt)), cpt)
    }

    saveTo(base / "chars.byfreq.dic", chars.map { case (s, c) => s"$s\t$c"})
    saveTo(base / "chars.bycode.dic", chars.sortBy(_._1).map { case (s, c) => s"$s\t$c"})
    saveTo(base / "chars.byfreq.vdic", CharDictionary.strings ++ chars.map(x => codeptRepr(x._1)))
    saveTo(base / "chars.bycode.vdic", CharDictionary.strings ++ chars.map(x => codeptRepr(x._1)).sorted)

    val pos = byType("pos").sortBy(-_._2)
    saveTo(base / "pos.dic", pos.map { case (s, c) => s"$s\t$c"})
    saveTo(base / "pos.vdic", "NUL" +: pos.map {_._1})

    val subpos = byType("subpos").sortBy(-_._2)
    saveTo(base / "subpos.dic", subpos.map { case (s, c) => s"$s\t$c"})
    saveTo(base / "subpos.vdic", "NUL" +: subpos.map {_._1})

    val ctype = byType("conjtype").sortBy(-_._2)
    saveTo(base / "ctype.dic", ctype.map { case (s, c) => s"$s\t$c"})
    saveTo(base / "ctype.vdic", "NUL" +: ctype.map {_._1})

    val cform = byType("conjform").sortBy(-_._2)
    saveTo(base / "cform.dic", cform.map { case (s, c) => s"$s\t$c"})
    saveTo(base / "cform.vdic", "NUL" +: cform.map {_._1})
  }

  private def process(parser: XJppLatticeParser, data: String) = {
    val item = parser.parse(new BufferedReader(new StringReader(data)))
    item.nodes.flatMap { n =>
      val chars = if (n.ranks.contains(1)) {
        val cpts = UnicodeUtil.stream(n.surface).map(x => new String(Character.toChars(x))).toSeq
        cpts.map("cpt" -> _)
      } else Nil

      val p = n.stringPos.get

      chars ++ Seq(
        "pos" -> p.pos,
        "subpos" -> p.subpos,
        "conjtype" -> p.conjType,
        "conjform" -> p.conjForm
      )
    }
  }

  def saveTo(path: Path, data: Iterable[String]): Unit = {
    path.writeLines(data)
  }

  def main(args: Array[String]): Unit = {
    val a = new Jpp2TagsArgs(args)
    a.verify()

    val filenames = a.input().lines().filter(_.length > 5).mkString(",")

    val conf = new SparkConf().setAppName("JumanppToTags")
    conf.registerKryoClasses(Array(
      classOf[Jpp2TagsArgs]
    ))

    val sc = new SparkContext(conf)

    try {
      doWork(sc, filenames, a.output(), a.maxChars())
    } finally {
      sc.stop()
    }
  }
}
