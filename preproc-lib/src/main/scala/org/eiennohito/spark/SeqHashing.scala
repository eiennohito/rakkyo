package org.eiennohito.spark

import scala.util.hashing.MurmurHash3

object SeqHashing {
  def hashSeq(seq: Seq[Long], seed: Int = 0xdeadbeef): Int = {
    var state = seed
    val iter = seq.iterator
    var len = 0
    while (iter.hasNext) {
      val a = iter.next()
      val aHigh = (a >>> 32).toInt
      val aLow = a.toInt
      state = MurmurHash3.mix(state, aHigh)
      state = MurmurHash3.mix(state, aLow)
      len += 1
    }
    MurmurHash3.mixLast(state, len)
  }
}
