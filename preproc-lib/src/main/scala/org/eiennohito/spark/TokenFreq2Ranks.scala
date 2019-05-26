package org.eiennohito.spark

import java.io.{OutputStreamWriter, PrintWriter}
import java.util

import ws.kotonoha.akane.io.Charsets

import scala.collection.mutable.ArrayBuffer

object TokenFreq2Ranks {
  import ws.kotonoha.akane.resources.FSPaths._
  def main(args: Array[String]): Unit = {
    val input = args(0)
    val output = args(1)
    val numTop = args(2).toInt
    val numBuckets = args(3).toInt

    val freqs = new ArrayBuffer[(String, Long)]()

    for {
      file <- input.p.children().filter(_.getFileName.toString.startsWith("part-")).toSeq.sorted
      line <- file.lines()
    } {
      val parts = line.split('\t')
      freqs += parts(0) -> parts(1).toLong
    }

    val tops = freqs.take(numTop).map(_._1)
    val rest = freqs.drop(numTop)

    val cumsum = new ArrayBuffer[(String, Long)]
    val xiter = rest.iterator
    var total = 0L
    while (xiter.hasNext) {
      val (s, cnt) = xiter.next()
      cumsum += s -> (cnt + total)
      total += cnt
    }

    val maxRank = numBuckets - 1
    val perRank = total.toDouble / maxRank
    val boundaries = (1 to numBuckets).map(_ * perRank).toArray

    for {
      os <- output.p.outputStream()
      writer <- new OutputStreamWriter(os, Charsets.utf8).res
    } {
      for (t <- tops) {
        writer.write(s"$t\t0\n")
      }
      var rawRank = 0
      for ((t, cnt) <- cumsum) {
        while (cnt >= boundaries(rawRank) && (rawRank + 1) < maxRank) {
          rawRank += 1
        }
        val rank = rawRank + 1
        writer.write(s"$t\t$rank\n")
      }
    }
  }
}
