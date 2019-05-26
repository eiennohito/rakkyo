package org.eiennohito.spark

import java.io.BufferedReader

import org.apache.commons.io.input.CharSequenceReader
import org.apache.commons.lang3.StringUtils

import scala.collection.mutable.ArrayBuffer

case class XMorphChar(
  char: String,
  tags: Array[(String, Float)]
)

case class XMorphDebugItem(
  comment: String,
  chars: IndexedSeq[XMorphChar]
)

object XMorphDebugParser {
  def parse(sent: CharSequence): XMorphDebugItem = {
    val reader = new CharSequenceReader(sent)
    val brdr = new BufferedReader(reader)
    parse(brdr)
  }

  def parse(in: BufferedReader): XMorphDebugItem = {
    var line = ""
    var comment = ""
    val data = new ArrayBuffer[XMorphChar]()
    while ({line = in.readLine(); line != null && line != "EOS"}) {
      if (line.startsWith("# ")) {
        comment = line
      } else {
        val parts = StringUtils.split(line, '\t')
        val char = parts(0)
        val ntags = (parts.length - 1) / 2
        var i = 1
        val bldr = Array.newBuilder[(String, Float)]
        while (i <= ntags) {
          val tag = parts(i)
          val tagProb = parts(i + ntags)
          bldr += tag -> tagProb.toFloat
          i += 1
        }
        data += XMorphChar(char, bldr.result())
      }
    }
    if (data.isEmpty && line == null) {
      return null
    }
    XMorphDebugItem(comment, data)
  }
}