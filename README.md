<div align=center>
 <img src="src/logo.png" width="180px">
</div>
<h2 align="center">
  <a href="">🔥🔥🔥 LatentCoT-Horizon</a>
</h2>
<p align="center">
  If you like our project, please give us a star ⭐ on GitHub for the latest update.
</p>
<p align="center">
  <a href="https://awesome.re">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
  <a href="https://github.com/multimodal-art-projection/LatentCoT-Horizon/pulls">
    <img src="https://img.shields.io/badge/PRs-Welcome-green.svg" alt="PRs Welcome">
  </a>
  <img src="https://img.shields.io/github/stars/multimodal-art-projection/LatentCoT-Horizon?logoColor=%23C8A2C8&amp;color=%23DCC6E0" alt="GitHub Repo stars">
</p>

This repository provides the papers mentioned in the survey "A Survey on Latent Reasoning".

## 📑 Citation

If you find our survey useful for your research, please consider citing the following paper:

```bibtex
@article{map2025latent,
  title={A Survey on Latent Reasoning},
  author={M-A-P},
  journal={arxiv},
  year={2025}
}
```

## 📣 Update News

`[2025-07-04]` We have initialed the repository.

## 💼 Contents

- [📑 Citation](#-citation)
- [📣 Update News](#-update-news)
- [💼 Contents](#-contents)
- [📜 Papers](#-papers)
  - [🧠 Latent CoT Reasoning](#-latent-cot-reasoning)
    - [🔄 Activation-based Recurrent Methods](#-activation-based-recurrent-methods)
      - [🧱 Architectural Recurrence](#-architectural-recurrence)
      - [🏋️ Training-induced Recurrence](#️-training-induced-recurrence)
      - [🎯 Training Strategies for Recurrent Reasoning](#-training-strategies-for-recurrent-reasoning)
      - [✨ Applications and Capabilities](#-applications-and-capabilities)
    - [⏳ Temporal Hidden-state Methods](#-temporal-hidden-state-methods)
      - [📦 Hidden-state based methods](#-hidden-state-based-methods)
      - [⚙️ Optimization-based State Evolution](#️-optimization-based-state-evolution)
      - [🎭 Training-induced Hidden-State Conversion](#-training-induced-hidden-state-conversion)
  - [🔬 Mechanistic Interpretability](#-mechanistic-interpretability)
    - [🧐 Do Layer Stacks Reflect Latent CoT?](#-do-layer-stacks-reflect-latent-cot)
    - [🛠️ Mechanisms of Latent CoT in Layer Representation](#️-mechanisms-of-latent-cot-in-layer-representation)
    - [💻 Turing Completeness of Layer-Based Latent CoT](#-turing-completeness-of-layer-based-latent-cot)
  - [♾️ Towards Infinite-depth Reasoning](#️-towards-infinite-depth-reasoning)
    - [🌀 Spatial Infinite Reasoning: Text Diffusion Models](#-spatial-infinite-reasoning-text-diffusion-models)
      - [⬛ Masked Diffusion Models](#-masked-diffusion-models)
      - [🔗 Chain-of-Thought Diffusion Models](#-chain-of-thought-diffusion-models)
      - [🧬 Hybrid Diffusion and Autoregressive Architectures](#-hybrid-diffusion-and-autoregressive-architectures)
    - [🕸️ Towards an 'Infinitely Long' Optimiser Network](#️-towards-an-infinitely-long-optimiser-network)
    - [📌 Implicit Fixed Point RNNs](#-implicit-fixed-point-rnns)
    - [💬 Discussion](#-discussion)
- [👍 Acknowledgement](#-acknowledgement)
- [♥️ Contributors](#️-contributors)

## 📜 Papers
### 🧠 Latent CoT Reasoning

#### 🔄 Activation-based Recurrent Methods

##### 🧱 Architectural Recurrence

| Title                                                        | Venue     | Date             | Links                                                        |
| ------------------------------------------------------------ |---------- | ---------------- | ------------------------------------------------------------ |
| **Universal transformers** | ICLR 2019 | Jul 2018 | [Paper](https://arxiv.org/abs/1807.03819) - [Code](https://github.com/andreamad8/Universal-Transformer-Pytorch) |
| **CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference** |ICLR 2025 | Oct 2023 | [Paper](https://arxiv.org/abs/2310.10845) |
| **AlgoFormer: An Efficient Transformer Framework with Algorithmic Structures** |TMLR 2025 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.13572) - [Code](https://github.com/chuanyang-Zheng/Algoformer)|
| **Relaxed recursive transformers: Effective parameter sharing with layer-wise Lora** |ICLR 2025 | Oct 2024 | [Paper](https://arxiv.org/abs/2410.20672) |
| **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach** |ICLR 2025 | Feb 2025 | [Paper](https://arxiv.org/abs/2502.05171) - [Code](https://github.com/seal-rg/recurrent-pretraining)|
| **Pretraining Language Models to Ponder in Continuous Space** |arXiv | May 2025 | [Paper](https://arxiv.org/abs/2505.20674) - [Code](https://github.com/LUMIA-Group/PonderingLM) |



##### 🏋️ Training-induced Recurrence
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Think before you speak: Training Language Models With Pause Tokens** | ICLR 2024 | Oct 2023 | [Paper](https://arxiv.org/abs/2310.02226) |
| **Guiding Language Model Reasoning with Planning Tokens** | CoLM 2024 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.06634) - [Code](https://github.com/Xinyi-Wang-dot/Planning-Tokens) |
| **Let's Think Dot by Dot: Hidden computation in transformer language models** | CoLM 2024 | May 2024 | [Paper](https://arxiv.org/abs/2405.04929) |
| **Disentangling memory and reasoning ability in large language models** | arXiv | Nov 2024 | [Paper](https://arxiv.org/abs/2411.13504) |
| **Training large language models to reason in a continuous latent space** | CoLM 2024 | Dec 2024 | [Paper](https://arxiv.org/abs/2412.06769) |
| **Compressed chain of thought: Efficient reasoning through dense representations** | arXiv | Dec 2024 | [Paper](https://arxiv.org/abs/2412.13171) |
| **Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.03275) - [Code](https://github.com/LUMIA-Group/Token-Assorted) |
| **Lightthinker: Thinking step-by-step compression** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.15589) - [Code](https://github.com/zjukg/LightThinker) |
| **Codi: Compressing chain-of-thought into continuous space via self-distillation** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.21074) |
| **System-1.5 Reasoning: Traversal in Language and Latent Spaces with Dynamic Shortcuts** | arXiv | May 2025 | [Paper](https://arxiv.org/abs/2505.18962) |
| **Parallel Continuous Chain-of-Thought with Jacobi Iteration** | arXiv | Jun 2025 | [Paper](https://arxiv.org/abs/2506.18582) |

##### 🎯 Training Strategies for Recurrent Reasoning

| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **From explicit cot to implicit cot: Learning to internalize cot step by step** | arXiv | May 2024 | [Paper](https://arxiv.org/abs/2405.14838) |
| **On the inductive bias of stacking towards improving reasoning** | NeurIPS 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.03149) |
| **Training large language models to reason in a continuous latent space** | CoLM 2024 | Dec 2024 | [Paper](https://arxiv.org/abs/2412.06769) |
| **Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.08482) |
| **Reasoning with latent thoughts: On the power of looped transformers** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.17416) |


##### ✨ Applications and Capabilities

| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks** | NeurIPS 2021 | Oct 2021 | [Paper](https://arxiv.org/abs/2110.11112) - [Code](https://github.com/tolga-ertugrul/learning-an-algorithm) |
| **Looped transformers as programmable computers** | ICML 2023 | Jun 2023 | [Paper](https://arxiv.org/abs/2306.08022) - [Code](https://github.com/giannou/looped-transformers) |
| **Simulation of graph algorithms with looped transformers** | arXiv | Feb 2024 | [Paper](https://arxiv.org/abs/2402.01107) - [Code](https://github.com/ADeLuca99/Looped-Transformers-on-Graphs) |
| **Guiding Language Model Reasoning with Planning Tokens** | CoLM 2024 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.06634) - [Code](https://github.com/Xinyi-Wang-dot/Planning-Tokens) |
| **Can looped transformers learn to implement multi-step gradient descent for in-context learning?** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.08292) |
| **Bypassing the exponential dependency: Looped transformers efficiently learn in-context by multi-step gradient descent** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.11268) |
| **Disentangling memory and reasoning ability in large language models** | arXiv | Nov 2024 | [Paper](https://arxiv.org/abs/2411.13504) |

#### ⏳ Temporal Hidden-state Methods

##### 📦 Hidden-state based methods
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Gated linear attention transformers with hardware-efficient training** | arXiv | Dec 2023 | [Paper](https://arxiv.org/abs/2312.06635) |
| **DeltaNet: A new scheme for parallelizing recurrent models over sequence length** | arXiv | Mar 2024 | [Paper](https://arxiv.org/abs/2403.04229) |
| **Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence** | arXiv | Apr 2024 | [Paper](https://arxiv.org/abs/2404.05892) - [Code](https://github.com/RWKV/RWKV-v5) |
| **Hgrn2: Gated linear rnns with state expansion** | arXiv | Apr 2024 | [Paper](https://arxiv.org/abs/2404.07904) - [Code](https://github.com/hgrn-stack/HGRN2) |
| **Transformers are ssms: Generalized models and efficient algorithms through structured state space duality** | arXiv | May 2024 | [Paper](https://arxiv.org/abs/2405.21060) - [Code](https://github.com/state-spaces/griffin) |
| **Parallelizing linear transformers with the delta rule over sequence length** | arXiv | Jun 2024 | [Paper](https://arxiv.org/abs/2406.06484) - [Code](https://github.com/yangsonglin/delta-transformer) |

##### ⚙️ Optimization-based State Evolution
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Transformers are rnns: Fast autoregressive transformers with linear attention** | ICML 2020 | Jun 2020 | [Paper](https://arxiv.org/abs/2006.16236) - [Code](https://github.com/idiap/fast-transformers) |
| **Retentive network: A successor to transformer for large language models** | arXiv | Jul 2023 | [Paper](https://arxiv.org/abs/2307.08621) - [Code](https://github.com/microsoft/unilm/tree/master/retnet) |
| **Gated linear attention transformers with hardware-efficient training** | arXiv | Dec 2023 | [Paper](https://arxiv.org/abs/2312.06635) |
| **Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence** | arXiv | Apr 2024 | [Paper](https://arxiv.org/abs/2404.05892) - [Code](https://github.com/RWKV/RWKV-v5) |
| **Hgrn2: Gated linear rnns with state expansion** | arXiv | Apr 2024 | [Paper](https://arxiv.org/abs/2404.07904) - [Code](https://github.com/hgrn-stack/HGRN2) |
| **Transformers are ssms: Generalized models and efficient algorithms through structured state space duality** | arXiv | May 2024 | [Paper](https://arxiv.org/abs/2405.21060) - [Code](https://github.com/state-spaces/griffin) |
| **Parallelizing linear transformers with the delta rule over sequence length** | arXiv | Jun 2024 | [Paper](https://arxiv.org/abs/2406.06484) - [Code](https://github.com/yangsonglin/delta-transformer) |
| **Learning to (learn at test time): Rnns with expressive hidden states** | arXiv | Jul 2024 | [Paper](https://arxiv.org/abs/2407.04620) |
| **Gated Delta Networks: Improving Mamba2 with Delta Rule** | arXiv | Dec 2024 | [Paper](https://arxiv.org/abs/2412.06464) |
| **Titans: Learning to memorize at test time** | arXiv | Jan 2025 | [Paper](https://arxiv.org/abs/2501.00663) |
| **Lattice: Learning to efficiently compress the memory** | arXiv | Apr 2025 | [Paper](https://arxiv.org/abs/2504.05646) |
| **It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization** | arXiv | Apr 2025 | [Paper](https://arxiv.org/abs/2504.13173) |
| **Atlas: Learning to optimally memorize the context at test time** | arXiv | May 2025 | [Paper](https://arxiv.org/abs/2505.23735) |
| **Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration** | arXiv | May 2025 | [Paper](https://arxiv.org/abs/2505.24688) |

##### 🎭 Training-induced Hidden-State Conversion
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Linearizing large language models** | arXiv | May 2024 | [Paper](https://arxiv.org/abs/2405.06640) |
| **Transformers to ssms: Distilling quadratic knowledge to subquadratic models** | NeurIPS 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.01289) |
| **LoLCATs: On Low-Rank Linearizing of Large Language Models** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.10254) |
| **Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.14458) - [Code](https://github.com/avivbick/llamba) |
| **Liger: Linearizing Large Language Models to Gated Recurrent Structures** | arXiv | Mar 2025 | [Paper](https://arxiv.org/abs/2503.01496) |

### 🔬 Mechanistic Interpretability

#### 🧐 Do Layer Stacks Reflect Latent CoT?
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Towards a mechanistic interpretation of multi-step reasoning capabilities of language models** | arXiv | Oct 2023 | [Paper](https://arxiv.org/abs/2310.14491) |
| **Iteration head: A mechanistic study of chain-of-thought** | NeurIPS 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.02546) - [Project](https://vivien-cabannes.github.io/iteration-head) |
| **Towards understanding how transformer perform multi-step reasoning with matching operation** | arXiv | Jun 2024 | [Paper](https://arxiv.org/abs/2406.04689) |
| **Do LLMs Really Think Step-by-step In Implicit Reasoning?** | arXiv | Nov 2024 | [Paper](https://arxiv.org/abs/2411.15862) |
| **Openai o1 system card** | arXiv | Dec 2024 | [Paper](https://arxiv.org/abs/2412.16720) |
| **Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning** | arXiv | Jan 2025 | [Paper](https://arxiv.org/abs/2501.12948) |
| **Back attention: Understanding and enhancing multi-hop reasoning in large language models** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.10835) |
| **How Do LLMs Perform Two-Hop Reasoning in Context?** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.13913) |
| **Reasoning with latent thoughts: On the power of looped transformers** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.17416) |
| **A little depth goes a long way: The expressive power of log-depth transformers** | arXiv | Mar 2025 | [Paper](https://arxiv.org/abs/2503.03961) |

#### 🛠️ Mechanisms of Latent CoT in Layer Representation
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting** | NeurIPS 2019 | Jul 2019 | [Paper](https://arxiv.org/abs/1907.00235) - [Code](https://github.com/ucla-ideas-lab/Transformer-XL) |
| **Transformer feed-forward layers are key-value memories** | EMNLP 2021 | Dec 2020 | [Paper](https://arxiv.org/abs/2012.14913) |
| **Interpretability in the wild: a circuit for indirect object identification in gpt-2 small** | arXiv | Nov 2022 | [Paper](https://arxiv.org/abs/2211.00593) - [Code](https://github.com/callaghang/ioi-circuit-extraction) |
| **micse: Mutual information contrastive learning for low-shot sentence embeddings** | arXiv | Nov 2022 | [Paper](https://arxiv.org/abs/2211.04928) - [Code](https://github.com/tassilo-klein/micse) |
| **How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model** | NeurIPS 2023 | May 2023 | [Paper](https://arxiv.org/abs/2305.00586) - [Code](https://github.com/michael-hanna/interp-greater-than) |
| **A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis** | EMNLP 2023 | May 2023 | [Paper](https://arxiv.org/abs/2305.15054) - [Code](https://github.com/Zurich-NLP/MECH-INTERP) |
| **Why lift so heavy? slimming large language models by cutting off the layers** | arXiv | Feb 2024 | [Paper](https://arxiv.org/abs/2402.11700) |
| **Do large language models latently perform multi-hop reasoning?** | EACL 2024 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.16837) |
| **How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning** | ICLR 2024 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.18312) |
| **The Unreasonable Ineffectiveness of the Deeper Layers** | arXiv | Mar 2024 | [Paper](https://arxiv.org/abs/2403.17887) |
| **Inheritune: Training Smaller Yet More Attentive Language Models** | arXiv | Apr 2024 | [Paper](https://arxiv.org/abs/2404.08634) - [Code](https://github.com/sandals-lab/inheritune) |
| **Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization** | ICML 2024 | May 2024 | [Paper](https://arxiv.org/abs/2405.15071) - [Code](https://github.com/boshijwang/grokking-reasoners) |
| **Loss landscape geometry reveals stagewise development of transformers** | Hi-DL 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.02452) |
| **Hopping too late: Exploring the limitations of large language models on multi-hop queries** | arXiv | Jun 2024 | [Paper](https://arxiv.org/abs/2406.12775) - [Code](https://github.com/biu-nlp/hopping-too-late) |
| **Distributional reasoning in llms: Parallel reasoning processes in multi-hop reasoning** | arXiv | Jun 2024 | [Paper](https://arxiv.org/abs/2406.13858) |
| **Unveiling Factual Recall Behaviors of Large Language Models through Knowledge Neurons** | arXiv | Aug 2024 | [Paper](https://arxiv.org/abs/2408.03247) - [Code](https://github.com/WadeWfy/Knowledge-Neurons) |
| **Unveiling induction heads: Provable training dynamics and feature learning in transformers** | arXiv | Sep 2024 | [Paper](https://arxiv.org/abs/2409.10559) |
| **Investigating layer importance in large language models** | arXiv | Sep 2024 | [Paper](https://arxiv.org/abs/2409.14381) |
| **Unifying and Verifying Mechanistic Interpretations: A Case Study with Group Operations** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.07476) - [Code](https://github.com/willy-wu/unified_circuits) |
| **Understanding Layer Significance in LLM Alignment** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.17875) |
| **Does representation matter? exploring intermediate layers in large language models** | arXiv | Dec 2024 | [Paper](https://arxiv.org/abs/2412.09563) |
| **Layer by Layer: Uncovering Hidden Representations in Language Models** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.02013) |
| **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.05171) - [Code](https://github.com/JonasGeiping/recurrent-pretraining) |
| **The Curse of Depth in Large Language Models** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.05795) - [Code](https://github.com/shweistein/the-curse-of-depth-in-llms) |
| **Back attention: Understanding and enhancing multi-hop reasoning in large language models** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.10835) |
| **The Representation and Recall of Interwoven Structured Knowledge in LLMs: A Geometric and Layered Analysis** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.10871) |
| **An explainable transformer circuit for compositional generalization** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.15801) |
| **Emergent Abilities in Large Language Models: A Survey** | arXiv | Mar 2025 | [Paper](https://arxiv.org/abs/2503.05788) |
| **Unpacking Robustness in Inflectional Languages: Adversarial Evaluation and Mechanistic Insights** | arXiv | May 2025 | [Paper](https://arxiv.org/abs/2505.07856) |
| **Do Language Models Use Their Depth Efficiently?** | arXiv | May 2025 | [Paper](https://arxiv.org/abs/2505.13898) - [Code](https://github.com/robertcsordas/depth_efficiency) |
| **Void in Language Models** | arXiv | May 2025 | [Paper](https://arxiv.org/abs/2505.14467) |

#### 💻 Turing Completeness of Layer-Based Latent CoT
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **An outsider's view of neural nets** | Cognitive Science | 1986 | [Paper](https://doi.org/10.1207/s15516709cog1001_1) |
| **Finding Structure in Time** | Cognitive Science | 1990 | [Paper](https://doi.org/10.1207/s15516709cog1402_1) |
| **On the computational power of neural nets** | JCSS | 1995 | [Paper](https://doi.org/10.1006/jcss.1995.1018) |
| **Long Short-Term Memory** | Neural Computation | 1997 | [Paper](https://doi.org/10.1162/neco.1997.9.8.1735) |
| **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation** | EMNLP 2014 | Jun 2014 | [Paper](https://arxiv.org/abs/1406.1078) |
| **Attention is all you need** | NeurIPS 2017 | Jun 2017 | [Paper](https://arxiv.org/abs/1706.03762) |
| **On the turing completeness of modern neural network architectures** | IJCNN 2021 | Jan 2019 | [Paper](https://arxiv.org/abs/1901.03429) |
| **Recurrent memory transformer** | NeurIPS 2022 | Jul 2022 | [Paper](https://arxiv.org/abs/2207.06881) - [Code](https://github.com/boildamage/Recurrent-Memory-Transformer) |
| **Looped transformers as programmable computers** | ICML 2023 | Jun 2023 | [Paper](https://arxiv.org/abs/2306.08022) - [Code](https://github.com/giannou/looped-transformers) |
| **On limitations of the transformer architecture** | CoLM 2024 | Nov 2023 | [Paper](https://arxiv.org/abs/2311.08107) |
| **Investigating Recurrent Transformers with Dynamic Halt** | arXiv | Feb 2024 | [Paper](https://arxiv.org/abs/2402.00976) |
| **Chain of thought empowers transformers to solve inherently serial problems** | ICLR 2024 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.12875) |
| **Quiet-star: Language models can teach themselves to think before speaking** | arXiv | Mar 2024 | [Paper](https://arxiv.org/abs/2403.09629) |
| **Ask, and it shall be given: On the Turing completeness of prompting** | arXiv | Nov 2024 | [Paper](https://arxiv.org/abs/2411.01992) |
| **Reinforcement Pre-Training** | arXiv | Jun 2025 | [Paper](https://arxiv.org/abs/2506.08007) |
| **Constant Bit-size Transformers Are Turing Complete** | arXiv | Jun 2025 | [Paper](https://arxiv.org/abs/2506.12027) |

### ♾️ Towards Infinite-depth Reasoning

#### 🌀 Spatial Infinite Reasoning: Text Diffusion Models

##### ⬛ Masked Diffusion Models
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Structured denoising diffusion models in discrete state-spaces** | NeurIPS 2021 | Jul 2021 | [Paper](https://arxiv.org/abs/2107.03006) |
| **Discrete diffusion modeling by estimating the ratios of the data distribution** | ICML 2024 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.04690) |
| **Your absorbing discrete diffusion secretly models the conditional distributions of clean data** | arXiv | Jun 2024 | [Paper](https://arxiv.org/abs/2406.03736) |
| **Learning Iterative Reasoning through Energy Diffusion** | ICML 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.05928) - [Project](https://yilundu.github.io/energy-diffusion/) |
| **Simplified and generalized masked diffusion for discrete data** | NeurIPS 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.18242) |
| **Simple and effective masked diffusion language models** | NeurIPS 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.19509) - [Code](https://github.com/ssahoo04/MDLM) |
| **Scaling up Masked Diffusion Models on Text** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.18514) |
| **TESS 2: A Large-Scale Generalist Diffusion Language Model** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.13917) |
| **Large Language Diffusion Models** | ICLR 2025 Workshop | Feb 2025 | [Paper](https://arxiv.org/abs/2502.14920) - [Project](https://llm-diffusion.github.io/) |

##### 🔗 Chain-of-Thought Diffusion Models
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models** | ICLR 2024 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.07754) - [Code](https://github.com/HKU-BALI/Diffusion-of-Thoughts) |
| **Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.14157) - [Code](https://github.com/HKU-BALI/D2P) |

##### 🧬 Hybrid Diffusion and Autoregressive Architectures
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models** | ICLR 2024 | Feb 2024 | [Paper](https://arxiv.org/abs/2402.07754) - [Code](https://github.com/HKU-BALI/Diffusion-of-Thoughts) |
| **Learning Iterative Reasoning through Energy Diffusion** | ICML 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.05928) - [Project](https://yilundu.github.io/energy-diffusion/) |
| **Simple and effective masked diffusion language models** | NeurIPS 2024 | Jun 2024 | [Paper](https://arxiv.org/abs/2406.19509) - [Code](https://github.com/ssahoo04/MDLM) |
| **Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.14157) - [Code](https://github.com/HKU-BALI/D2P) |
| **Dream 7B: A Diffusion-based Autoregressive Model for Text Generation** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.15539) - [Project](https://hkunlp.github.io/blog/2025/dream) |
| **Scaling Diffusion Language Models via Adaptation from Autoregressive Models** | ICLR 2025 | Oct 2024 | [Paper](https://arxiv.org/abs/2410.18433) |
| **Scaling up Masked Diffusion Models on Text** | arXiv | Oct 2024 | [Paper](https://arxiv.org/abs/2410.18514) |
| **Large Language Models to Diffusion Finetuning** | arXiv | Jan 2025 | [Paper](https://arxiv.org/abs/2501.15781) |
| **TESS 2: A Large-Scale Generalist Diffusion Language Model** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.13917) |
| **Large Language Diffusion Models** | ICLR 2025 Workshop | Feb 2025 | [Paper](https://arxiv.org/abs/2502.14920) - [Project](https://llm-diffusion.github.io/) |
| **Mercury: Ultra-Fast Language Models Based on Diffusion** | arXiv | Jun 2025 | [Paper](https://arxiv.org/abs/2506.18206) |
| **Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities** | Technical Report | Jun 2025 | [Paper](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) |


#### 🕸️ Towards an 'Infinitely Long' Optimiser Network
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Leave no context behind: Efficient infinite context transformers with infini-attention** | arXiv | Apr 2024 | [Paper](https://arxiv.org/abs/2404.07143) - [Project](https://sites.google.com/view/infini-attention) |
| **Learning to (learn at test time): Rnns with expressive hidden states** | arXiv | Jul 2024 | [Paper](https://arxiv.org/abs/2407.04620) |
| **Titans: Learning to memorize at test time** | arXiv | Jan 2025 | [Paper](https://arxiv.org/abs/2501.00663) |
| **Atlas: Learning to optimally memorize the context at test time** | arXiv | May 2025 | [Paper](https://arxiv.org/abs/2505.23735) |

#### 📌 Implicit Fixed Point RNNs
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **Implicit Language Models are RNNs: Balancing Parallelization and Expressivity** | arXiv | Feb 2025 | [Paper](https://arxiv.org/abs/2502.07827) |

#### 💬 Discussion
| Title | Venue | Date | Links |
| --- | --- | --- | --- |
| **A survey of diffusion models in natural language processing** | TACL | May 2023 | [Paper](https://arxiv.org/abs/2305.14671) |
| **Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis** | arXiv | Dec 2024 | [Paper](https://arxiv.org/abs/2412.04431) |
| **Large Language Diffusion Models** | ICLR 2025 Workshop | Feb 2025 | [Paper](https://arxiv.org/abs/2502.14920) - [Project](https://llm-diffusion.github.io/) |

## 👍 Acknowledgement
- [Awesome-Latent-CoT](https://github.com/EIT-NLP/Awesome-Latent-CoT): a curated list of papers exploring latent chain-of-thought reasoning in large language models.  ￼

## ♥️ Contributors

<a href="https://github.com/multimodal-art-projection/LatentCoT-Horizon/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=multimodal-art-projection/LatentCoT-Horizon" />
</a>


