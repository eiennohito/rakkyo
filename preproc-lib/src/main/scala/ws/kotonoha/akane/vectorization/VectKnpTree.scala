package ws.kotonoha.akane.vectorization

import ws.kotonoha.akane.analyzers.knp.LexemeApi

/**
  * @author eiennohito
  * @since 2017/02/09
  */
object VectKnpTree {
  def feature(kind: String): LexemeApi => String = {
    kind match {
      case "norm" => (api) => api.valueOfFeature("正規化代表表記").getOrElse(api.canonicForm())
      case "surf" => (api) => api.surface
      case _ => (api) => api.canonicForm()
    }
  }
}
