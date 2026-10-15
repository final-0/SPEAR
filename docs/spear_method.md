# SPEAR — Method

\section{SPEAR: Staged Pre-training and Expert Adapter Routing}
SPEAR is a staged learning framework for generalizable optical flow estimation across diverse visual conditions, ranging from clean images to severely corrupted observations.
The framework first establishes transferable flow representations and subsequently cultivates domain-specialized adapters through expert adapter routing to enhance robustness.
In the following, we describe these two stages of SPEAR.

\subsection{Pre-training for Generalizable Flow Representations}
Leveraging representations extracted from large-scale image encoders has become a standard paradigm across a wide range of vision tasks.
The effectiveness of such transfer-based approaches largely depends on the quality of the underlying feature extractor, where stronger and more diverse pre-training generally leads to improved downstream performance.
Following this observation, we initialize the feature encoder from a large-scale pre-trained vision model and adapt it to optical flow estimation through supervised flow learning.
This stage focuses on acquiring transferable flow representations that serve as the basis for the subsequent robustness-specialization stage.

A suitable architecture for validating this hypothesis is WAFT, which leverages a pre-trained Depth Anything V2-Small encoder for optical flow estimation.
Building upon this design, we adopt larger encoder variants, namely Depth Anything V2-Large (the configuration in this repository) and -Giant, and transfer their representations to the flow estimation network through a ResNet-based projection module.
The projected features are subsequently refined by a recurrent update operator.

This constitutes the base architecture of SPEAR. However, effectively transferring the significantly richer representations produced by large-scale encoders to the optical flow domain is non-trivial.
The feature dimensionality and semantic richness increase substantially with encoder scale, introducing a larger representation gap between the visual backbone and the flow estimation network.
To facilitate stable transfer, we initialize the recurrent update operator, the flow heads and the ResNet image-feature network (except its output layer) with the weights of a pre-trained WAFT model.
In contrast, the DPT head on the large encoder and the layer fusing its output with the ResNet features are randomly initialized and optimized during flow pre-training.
This initialization strategy allows the model to inherit motion estimation capabilities from WAFT while gradually adapting to the richer visual representations provided by larger foundation encoders.

\subsection{Refining Robustness through Expert Adapter Routing}
While large-scale pre-training provides strong and transferable flow representations, a single model remains insufficient to handle the diverse appearance shifts induced by adverse conditions.
A straightforward approach is to fine-tune the entire network on corrupted data. However, such adaptation risks overwriting the general representations acquired during pre-training and entangles robustness learning with flow representation learning.
Instead, SPEAR decouples these objectives by keeping the pre-trained flow estimator entirely frozen and introducing lightweight expert adapters for robustness specialization.
As a result, the transferable flow representations learned during the first stage are preserved, while robustness is acquired independently through a small number of newly introduced parameters.

The expert adapters are inserted as residual branches on the projected feature map of each input frame, and their activation is controlled by a lightweight token router built upon the semantic representations extracted by the foundation encoder.
Let $I$ denote an input frame and $E_{\phi}$ the frozen foundation encoder with parameters $\phi$.
The encoder partitions $I \in \mathbb{R}^{H \times W \times 3}$ into $N = HW/p^{2}$ patches of size $p \times p$ and, at each intermediate layer $\ell \in \mathcal{L}$, produces a set of $d$-dimensional tokens,
\begin{equation}
\{\mathbf{t}^{(\ell)}_{i}\}_{i=1}^{N} = E_{\phi}^{(\ell)}(I), \qquad \mathbf{t}^{(\ell)}_{i} \in \mathbb{R}^{d}.
\end{equation}
Crucially, the \emph{same} frozen tokens feed both the flow branch and the router, so that corruption-aware routing requires no additional feature-extraction backbone:
\begin{equation}
\mathbf{F} = \mathrm{Proj}_{\theta}\!\big(\{\mathbf{t}^{(\ell)}_{i}\}_{\ell \in \mathcal{L}}\big),
\qquad
\mathbf{r} = \mathrm{Pool}\!\big(\{\mathbf{t}^{(\ell)}_{i}\}\big),
\qquad
\boldsymbol{\alpha} = \sigma\!\big(\mathbf{W}_{r}\mathbf{r} + \mathbf{b}_{r}\big),
\end{equation}
where $\mathbf{F}$ is the projected flow feature, $\mathrm{Pool}$ aggregates the tokens into a routing descriptor $\mathbf{r}$, and $\boldsymbol{\alpha} \in [0,1]^{K}$ are the gating signals for the $K$ experts.
These gates modulate the residual outputs of the expert adapters $\{\mathcal{A}_{k}\}_{k=1}^{K}$, which are applied in sequence and whose output projections are zero-initialized, so that the specialized model coincides with the frozen estimator at the onset of training,
\begin{equation}
\mathbf{F}_{0} = \mathbf{F}, \qquad
\mathbf{F}_{k} = \mathbf{F}_{k-1} + \alpha_{k}\, \mathcal{A}_{k}(\mathbf{F}_{k-1}), \qquad
\hat{\mathbf{F}} = \mathbf{F}_{K}.
\end{equation}
Consequently, robustness is acquired additively: the network departs from its clean behavior only where an expert is activated, and on clean inputs the gates collapse toward zero so that $\hat{\mathbf{F}} \to \mathbf{F}$.
We use $K = 3$ expert groups: frost; the other weather corruptions (fog, spatter, snow, rain); and compression-type corruptions (JPEG, pixelation, elastic transform). The remaining corruptions are assigned to no expert, and the router is trained to keep all gates closed for them and for clean input.

In this stage, the router is trained using the corruption labels associated with the synthetic degradations employed for data augmentation.
Rather than introducing an additional classification backbone, the router directly utilizes the encoder tokens, which already capture rich scene semantics and image characteristics.
This design enables corruption-aware adaptation while maintaining a shared and frozen SPEAR backbone.
Because routing is performed on top of foundation-level representations, the framework naturally supports different numbers of experts without modifying the underlying flow estimator.

During robustness specialization, we further extend the conventional two-frame formulation to a three-frame input setting.
The additional temporal observation provides complementary visual evidence that is often unavailable in a single frame pair, improving correspondence estimation when individual observations are partially degraded by corruption.
While the token router gates the corruption-specific appearance experts, this temporal context is incorporated through two gated temporal adapters (one initializing the hidden state of each direction from the whole triplet, one exchanging information between the forward and backward recurrent states), which are not routed and thus remain available throughout. Their outputs are also zero-initialized.

\section{Implementation and Training Details}
\paragraph{Staged optimization.}
The feature encoder is kept frozen throughout, and only the flow-estimation components are optimized.
Pre-training proceeds over three datasets in increasing order of domain proximity to the target, namely TartanAir, FlyingThings3D, and Spring.
On TartanAir and FlyingThings3D we adopt a \emph{refine-freeze} schedule: the recurrent update operator and the flow heads, initialized from WAFT, are held fixed, and only the encoder-interface modules (DPT head, ResNet image-feature network and the fusion layer) are trained, so that the network learns to align the large encoder's features with the pre-existing motion prior.
In the final Spring stage all non-encoder parameters are unfrozen and fine-tuned jointly at the native $1080 \times 1920$ resolution, allowing the whole flow network to specialize to the target domain.

\paragraph{Warm-start by shape-compatible loading.}
Because the projection and fusion modules change shape with encoder scale, the WAFT checkpoint is loaded partially: parameters whose shapes match (the recurrent operator, the flow heads and most of the ResNet image-feature network) are inherited, whereas the encoder-interface layers (DPT head, fusion layer, ResNet output layer) are re-initialized and learned.
The encoder weights themselves are provided by the corresponding Depth Anything V2 checkpoint and are never overwritten by this warm-start.

\paragraph{Optimization.}
Pre-training uses AdamW (weight decay $10^{-5}$), bf16 mixed precision, gradient clipping at 1.0, and a one-cycle learning-rate schedule with linear warm-up over the first 5\% of the steps and linear decay.
The peak learning rate is $2\times10^{-4}$ for the first TartanAir stage (100k steps), $1\times10^{-4}$ for the second TartanAir stage (200k steps) and for FlyingThings3D (100k steps), and $5\times10^{-5}$ for Spring (20k steps).
The recurrent operator is unrolled for five refinement iterations at both training and inference.

\paragraph{Distributed setup and memory.}
Training is distributed with DistributedDataParallel on NVIDIA A100 (40\,GB) GPUs: four GPUs with a global batch of four for TartanAir ($480 \times 640$ crops), eight GPUs with a global batch of sixteen for FlyingThings3D ($400 \times 720$ crops), and eight GPUs with a global batch of eight, i.e.\ one sample per GPU, for Spring at $1080 \times 1920$.
During robustness specialization the frozen encoder runs under \texttt{no\_grad}, and activation checkpointing is applied to the adapters and to every recurrent iteration, so that three-frame training at native resolution fits within the 40\,GB budget.

\paragraph{Robustness specialization.}
The expert adapters and the token router are optimized in two phases with the base flow network frozen: the router is first trained for 10k steps with a binary cross-entropy loss to predict the expert-group label of each triplet from the encoder tokens, after which the zero-initialized adapters are optimized for 5k steps while the router is fixed.
Both phases use AdamW with a learning rate of $10^{-4}$, constant after a 100-step warm-up, and a global batch of eight native-resolution triplets.
Every training triplet is used twice. On the clean triplet, the loss is the flow likelihood loss plus a clean-regret term that penalizes only the pixels where the three-frame model is less accurate than the frozen two-frame parent; on a corrupted copy of the same triplet, the loss is the flow likelihood loss.
Training data are synthesized by applying benchmark-style corruptions to clean triplets: 25\% of the samples stay clean, 50\% receive a weather corruption, and 25\% one of the other fifteen corruptions.
Corruption severities are calibrated so that the pixel-level magnitude of each synthetic corruption matches that of the publicly available benchmark, and the weather corruptions (frost, rain, and snow) are reproduced from physically extracted overlays---static for frost and independently sampled per frame for rain and snow---consistent with their temporal statistics measured on the benchmark.
