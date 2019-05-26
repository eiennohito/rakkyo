package org.eiennohito.spark

class VectorizerKeyer(min: Int, max: Int) {

  def inRange(len: Int): Boolean = {
    (min <= len) && (len <= max)
  }

  def keyFor(p1: Int, p2: Int): (Int, Int) = {
    val reversed = max - p1
    val sign = p2 >>> 31
    if (sign != 0) {
      (reversed, p2)
    } else (-reversed, p2)
  }
}
