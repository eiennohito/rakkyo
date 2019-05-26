lazy val scalaPbVersion = "0.7.0"

(scalaVersion in ThisBuild) := "2.11.9"

def pbScala(grpc: Boolean = false): Seq[Setting[_]] = {
  Def.settings(
    PB.targets in Compile := Seq(
      scalapb.gen(grpc = grpc) -> (sourceManaged in Compile).value
    ),
    PB.protocVersion := "-v351",
    libraryDependencies ++= Seq(
      "com.thesamet.scalapb" %% "scalapb-runtime" % scalaPbVersion % "protobuf",
      "com.google.protobuf" % "protobuf-java" % "3.1.0"
    )
  )
}

def protoIncludes(files: Project*) = {
  val paths = files.map(f => f.base / "src" / "main" / "protobuf")
  Seq(PB.includePaths in Compile ++= paths)
}

lazy val commonSettings = Def.settings(
  (scalaVersion in ThisBuild) := "2.11.9",
  libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % Test
)

//imported project -- akane
lazy val akane = project

lazy val `akane-macros` =
  (project in file("akane/macros"))
    .settings(commonSettings)

lazy val `akane-util` =
  (project in file("akane/util"))
    .settings(commonSettings)

lazy val `akane-knp` = (project in file("akane/knp"))
  .settings(commonSettings, pbScala())

lazy val sparkCore = Def.settings(
  libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.1" % Provided
)

lazy val preproc = (project in file("preproc"))
    .dependsOn(`tensorflow-spark`, preprocLib, `akane-knp`)
    .settings(
      commonSettings,
      sparkCore
    )
  .settings(
    libraryDependencies += "org.rogach" %% "scallop" % "3.1.5",
    fork in Test := true,
    parallelExecution in Test := false,
    assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
  )

lazy val `tensorflow-example` = (project in file("scalapb-tensorflow/example"))
  .settings(pbScala(), commonSettings)

lazy val `tensorflow-spark` = (project in file("scalapb-tensorflow/spark"))
  .settings(pbScala(), commonSettings, sparkCore)
  .dependsOn(`tensorflow-example`)

lazy val preprocLib = (project in file("preproc-lib"))
  .settings(commonSettings)
  .dependsOn(`tensorflow-example`, `akane-knp`)

lazy val root = (project in file(".")).aggregate(preproc)