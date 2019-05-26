package org.eiennohito.spark

import java.io.OutputStreamWriter
import java.net.URI
import java.nio.file.{Path, Paths}

import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.AccumulatorV2
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.rogach.scallop.ScallopConf
import org.tensorflow.example.example.Example
import ws.kotonoha.akane.io.Charsets
import ws.kotonoha.akane.vectorization.{CharDictionary, SentenceDifficultyEstimator, TagDictionary, XMorphConverter}
import ws.kotonoha.spark.tensorflow.TFRecords

import scala.collection.mutable


class MapAccumulator[T] extends AccumulatorV2[T, Map[T, Long]] {
  private val innerMap = new mutable.HashMap[T, Long]()
  override def isZero: Boolean = innerMap.isEmpty

  override def copy(): AccumulatorV2[T, Map[T, Long]] = {
    val other = new MapAccumulator[T]
    other.innerMap ++= innerMap
    other
  }

  override def reset(): Unit = {
    innerMap.clear()
  }

  override def add(v: T): Unit = {
    val cur = innerMap.getOrElse(v, 0L)
    innerMap.update(v, cur + 1)
  }

  override def merge(other: AccumulatorV2[T, Map[T, Long]]): Unit = {
    other match {
      case ma: MapAccumulator[T] =>
        val oiter = ma.innerMap.iterator
        while (oiter.hasNext) {
          val (k, v) = oiter.next()
          val cur = innerMap.getOrElse(k, 0L)
          innerMap.update(k, cur + v)
        }
      case _ => throw new RuntimeException("Invalid map")
    }
  }

  override def value: Map[T, Long] = innerMap.toMap
}

object JumanppToXMorph {
  import ws.kotonoha.akane.resources.FSPaths._

  private class Jpp2MorphArgs(args: Seq[String]) extends ScallopConf(args) {
    val input = opt[String](required = true)
    val dicts = opt[Path](required = true)
    val output = opt[String](required = true)
    val stats = opt[String](default = output.toOption)
    val outputFiles = opt[Int](default = Some(1000))
    val maxLength = opt[Int](default = Some(100))
    val sortByLength = toggle(default = Some(false))
    val unkSymbolProb = opt[Float](default = Some(0))
    val zen2HanProb = opt[Float](default = Some(0))
    val diffDict = opt[Path]()
    val sampleRatio = opt[Double](default = Some(0.2))
    val boundaryRatio = opt[Double](default = Some(0.1))
    val storageLevel = opt[String](default = Some("DISK_ONLY")).map(StorageLevel.fromString)
    val topOne = toggle(default = Some(false))
    val bert = toggle(default = Some(true))
  }


  def doWork(sc: SparkContext, filenames: String, chars: CharDictionary,
    tagDicts: IndexedSeq[TagDictionary], args: Jpp2MorphArgs): Unit = {
    val data = EosDelimited.readFrom(sc, filenames)

    val charBcast = sc.broadcast(chars)
    val tagBcast = sc.broadcast(tagDicts)
    var diffDict: SentenceDifficultyEstimator = null
    if (args.diffDict.isSupplied) {
      println("loading difficulty dictionary...")
      diffDict = SentenceDifficultyEstimator.loadFromFile(args.diffDict(), "\u0001")
      println(s"Difficulty dictionary loaded: $diffDict")
    }

    val ddBcast = sc.broadcast(diffDict)
    val maxLength = args.maxLength()
    val unkSymProb = args.unkSymbolProb()
    val zen2HanProb = args.zen2HanProb()
    val topOne = args.topOne()
    val bert = args.bert()

    val items = data.mapPartitions(iter => {
      val worker = new XMorphConverter(
        charBcast.value, tagBcast.value, maxLength, unkSymProb, zen2HanProb, ddBcast.value, topOne, bert
      )
      iter.flatMap { txt =>
        worker.handle(txt.toString)
      }
    }).persist(args.storageLevel())

    val haveDict = args.diffDict.isSupplied

    val origAccumulator = new MapAccumulator[Int]
    sc.register(origAccumulator, "original")

    val sampledAcc = new MapAccumulator[Int]
    sc.register(sampledAcc, "sampled")

    val processed = if (haveDict) {
      items.foreach { case ((diff, _), _) => origAccumulator.add(diff) }
      val diffFreqs = origAccumulator.value

      val total = diffFreqs.values.sum
      val freq0 = diffFreqs(0)
      val maxSymbol = diffDict.default
      val freqMax = diffFreqs(maxSymbol)

      val fullBudget = total * args.sampleRatio()
      val numFixed = (freq0 + freqMax) * args.boundaryRatio()

      val remBudget = fullBudget - numFixed
      val remTotal = total - (freq0 + freqMax)
      val numInner = maxSymbol - 1
      val weightSeq = if (remBudget > remTotal) {
        val borderBudget = fullBudget - remTotal
        val borderRatio = borderBudget / (freq0 + freqMax)
        val inner = Seq.fill(numInner)(1.0)
        borderRatio +: inner :+ borderRatio
      } else {
        val ratio = remBudget / remTotal
        val baseC = ratio / 2
        val minC = 3 * ratio - 2
        val bias = baseC max minC
        val innerWeights = (0 until numInner).map { idx =>
          // smoothing function
          // was found as 2nd order polynomial satisfying:
          // 1) F(1) - F(0) = ratio
          // 2) f'(0.5) = 0
          // 3) f(x) <= 1
          // 4) f(0) = f(1) = bias
          val x = idx.toDouble / numInner
          3 * x * x * (2 * bias - 2 * ratio) + 2 * x * (-3 * bias + 3 * ratio) + bias
        }

        val outer = args.boundaryRatio()
        outer +: innerWeights :+ outer
      }
      val weights = weightSeq.map(_.toFloat).toArray

      args.stats().p.parent.foreach(_.mkdirs())

      for {
        os <- (args.stats() + ".samplestats").p.outputStream()
        wr <- new OutputStreamWriter(os, Charsets.utf8).res
      } {

        wr.write(s"total = $total\n")
        wr.write(s"borders = ${freq0 + freqMax}\n")
        wr.write(s"full budget = $fullBudget (${args.sampleRatio()})\n")
        wr.write(s"num fixed = $numFixed\n")
        wr.write(s"remaining budget = $remBudget (${remBudget / fullBudget})\n")
        wr.write(s"remaining total = $remTotal (${remTotal.toDouble / total})\n")
        wr.write(s"rho = ${remBudget / remTotal}\n")

        for (i <- weights.indices) {
          val num = diffFreqs(i)
          val ratio = num.toDouble / total * 100
          wr.write(f"$i\t${weights(i) * 100}%.3f\t$ratio%.3f\t$num\n")
        }
      }

      val weightBc = sc.broadcast(weights)
      items.filter {
        case ((diff, hash), _) =>
          // use hash instead of random for sampling
          val border = weightBc.value(diff)
          val hashPart = (hash & 0x3fffff00) >> 8
          val x = hashPart / 0x3fffff.toFloat
          val isSampled = x < border
          if (isSampled) {
            sampledAcc.add(diff)
          }
          isSampled
      }
    } else items

    diffDict = null

    val outFiles = args.outputFiles()

    val sorted = if (args.sortByLength()) {
      processed.persist(args.storageLevel()).sortByKey(numPartitions = outFiles)
    } else {
      processed.map { case ((b, a), v) => ((a, b), v) }.partitionBy(new HashPartitioner(outFiles))
    }

    TFRecords.writeToFile(sorted.values, args.output() + "/data", Some(classOf[GzipCodec]))

    if (haveDict) {
      for {
        os <- (args.stats() + ".samplereport").p.outputStream()
        wr <- new OutputStreamWriter(os, Charsets.utf8).res
      } {
        val origFreqs = origAccumulator.value
        val sampleFreqs = sampledAcc.value

        val max = origFreqs.keys.max

        val totalFreq = origFreqs.values.sum
        val totalSampled = sampleFreqs.values.sum

        for (i <- 0 to max) {
          val freq = origFreqs.getOrElse(i, 1L)
          val sampled = sampleFreqs.getOrElse(i, 1L)
          val samplPerc = sampled.toDouble / freq * 100
          val oldPerc = freq.toDouble / totalFreq * 100
          val newPerc = sampled.toDouble / totalSampled * 100
          wr.append(f"$i\t$samplPerc%.2f\t$oldPerc%.2f\t$newPerc%.2f\t$freq\t$sampled\n")
        }

        val perc = totalSampled.toDouble / totalFreq * 100
        wr.append(f"Total: $totalSampled/$totalFreq ($perc%.2f%%)")
      }
    }
  }

  def resolveInput(in: String): String = {
    val char = in.charAt(0)
    if (char == '@') {
      val lines = in.p.lines().filter(l => !l.startsWith("#") && l.length > 4)
      lines.mkString(",")
    } else {
      in
    }
  }

  def main(args: Array[String]): Unit = {
    val a = new Jpp2MorphArgs(args)
    a.verify()

    val filenames = resolveInput(a.input())

    val conf = new SparkConf().setAppName("Jumanpp2Rakkyo")
    conf.registerKryoClasses(Array(
      classOf[Jpp2MorphArgs],
      classOf[CharDictionary],
      classOf[TagDictionary],
      classOf[Example],
      classOf[SentenceDifficultyEstimator]
    ))

    val sc = new SparkContext(conf)

    val dicBase = a.dicts()
    val chardic = CharDictionary.load(dicBase / "chars.bycode.dic")

    val tagDicNames = IndexedSeq(
      "seg.dic",
      "pos.dic",
      "subpos.dic",
      "ctype.dic",
      "cform.dic"
    )

    val tagDicts = tagDicNames.map(nm => TagDictionary.load(dicBase / nm))

    try {
      doWork(sc, filenames, chardic, tagDicts, a)
    } finally {
      sc.stop()
    }
  }
}
