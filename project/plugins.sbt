addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.8")

val scalaPbVersion = "0.7.0"

addSbtPlugin("com.thesamet" % "sbt-protoc" % "0.99.17")

libraryDependencies += "com.thesamet.scalapb" %% "compilerplugin" % scalaPbVersion

libraryDependencies += "com.github.os72" % "protoc-jar" % "3.6.0"
