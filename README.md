# LangSpace_mapping

Abstract:

Accurate 3D scene understanding is essential for
robotics and augmented reality (AR), where high-quality
instance segmentation and semantic scene graphs enable
downstream reasoning and interaction. While recent meth-
ods such as ConceptGraphs [4] leverage vision-language
models (VLMs) and large language models (LLMs) to seg-
ment RGB-D sequences and build open-vocabulary scene
graphs, they are limited by incomplete viewpoint cover-
age, resulting in partial object reconstructions. This paper
proposes a complementary approach that integrates prior
knowledge in the form of known 3D object models to refine
and complete partial reconstructions. The method identifies
candidate object segments using semantic similarity from
CLIP [9] embeddings and aligns reference objects via ro-
bust geometric registration pipelines based on FPFH [10]
or PREDATOR [5] features, followed by RANSAC [3] and
ICP [13]. Integrated into the ConceptGraphs pipeline, the
approach shows improved global and per-object segmen-
tation accuracy on the Replica [11] dataset, particularly
for large and partially observed objects. This work demon-
strates the effectiveness of incorporating object-level priors
for more complete and accurate 3D scene representations,
and lays the groundwork for injecting instance-specific se-
mantics and affordances into scene graphs.

Check LangSpace_mapping.pdf for more details.
