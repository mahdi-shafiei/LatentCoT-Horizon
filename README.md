# LatentCOT-Horizon


<div align="center">
 <p align="center">

<!-- <a href="">📝 Paper</a> | <a href="#-papers">📄 List</a>  -->

 </p>
</div>
<div align="center">

<!-- [![LICENSE](https://img.shields.io/github/license/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling/blob/main/LICENSE)
![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
[![commit](https://img.shields.io/github/last-commit/Xnhyacinth/Long_Text_Modeling_Papers?color=blue)](https://github.com/Xnhyacinth/Long_Text_Modeling_Papers/commits/main)
[![PR](https://img.shields.io/badge/PRs-Welcome-red)](https://github.com/Xnhyacinth/Long_Text_Modeling_Papers/pulls)
[![GitHub Repo stars](https://img.shields.io/github/stars/Xnhyacinth/Awesome-LLM-Long-Context-Modeling)](https://github.com/Xnhyacinth/Awesome-LLM-Long-Context-Modeling) -->

</div>

This repository provides the papers mentioned in the survey "A Survey on Latent Reasoning".

If you find our survey useful for your research, please consider citing the following paper:

```bibtex
@article{map2025latent,
  title={A Survey on Latent Reasoning},
  author={M-A-P},
  journal={arxiv},
  year={2025}
}
```

## Contents

  - [📜 Papers](#-papers)
      - [Latent CoT Reasoning](#latent-cot-reasoning)
          - [Activation-based Recurrent Methods](#activation-based-recurrent-methods)
              - [Architectural Recurrence](#architectural-recurrence)
              - [Training-induced Recurrence](#training-induced-recurrence)
              - [Training Strategies for Recurrent Reasoning](#training-strategies-for-recurrent-reasoning)
              - [Applications and Capabilities](#applications-and-capabilities)
          - [Hidden-state Methods](#hidden-state-methods)
              - [Hidden-state based methods](#hidden-state-based-methods)
              - [Optimization-based State Evolution](#optimization-based-state-evolution)
              - [Training-induced Hidden-State Conversion](#training-induced-hidden-state-conversion)
      - [Mechanistic Interpretability](#mechanistic-interpretability)
          - [Do Layer Stacks Reflect Latent CoT?](#do-layer-stacks-reflect-latent-cot)
          - [Mechanisms of Latent CoT in Layer Representation](#mechanisms-of-latent-cot-in-layer-representation)
          - [Turing Completeness of Layer-Based Latent CoT](#turing-completeness-of-layer-based-latent-cot)
      - [Towards Infinite-depth Reasoning](#towards-infinite-depth-reasoning)
          - [Spatial Infinite Reasoning: Text Diffusion Models](#spatial-infinite-reasoning-text-diffusion-models)
              - [Masked Diffusion Models](#masked-diffusion-models)
              - [Chain-of-Thought Diffusion Models](#chain-of-thought-diffusion-models)
              - [Hybrid Diffusion and Autoregressive Architectures](#hybrid-diffusion-and-autoregressive-architectures)
          - [Towards an 'Infinitely Long' Optimiser Network](#towards-an-infinitely-long-optimiser-network)
          - [Implicit Fixed-Point RNNs](#implicit-fixed-point-rnns)
          - [Discussion](#57-discussion)


## 📜 Papers
### Latent CoT Reasoning

#### Activation-based Recurrent Methods

##### Architectural Recurrence
1.  [**Universal transformers.**](https://arxiv.org/abs/1807.03819) *Dehghani, Mostafa, Gouws, Stephan, Vinyals, Oriol, Uszkoreit, Jakob, and Kaiser, Lukasz.* arXiv preprint arXiv:1807.03819 2018.
2.  [**CoTFormer: A Chain-of-Thought Driven Architecture with Budget-Adaptive Computation Cost at Inference.**](https://arxiv.org/abs/2310.10845) *Mohtashami, Amirkeivan, Pagliardini, Matteo, and Jaggi, Martin.* arXiv preprint arXiv:2310.10845 2023.
3.  [**Relaxed recursive transformers: Effective parameter sharing with layer-wise lora.**](https://arxiv.org/abs/2410.20672) *Bae, Sangmin, Fisch, Adam, Harutyunyan, Hrayr, Ji, Ziwei, Kim, Seungyeon, and Schuster, Tal.* arXiv preprint arXiv:2410.20672 2024.
4.  [**AlgoFormer: An Efficient Transformer Framework with Algorithmic Structures.**](https://arxiv.org/abs/2402.13572) *Gao, Yihang, Zheng, Chuanyang, Xie, Enze, Shi, Han, Hu, Tianyang, Li, Yu, Ng, Michael K, Li, Zhenguo, and Liu, Zhaoqiang.* arXiv preprint arXiv:2402.13572 2024.
5.  [**Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.**](https://arxiv.org/abs/2502.05171) *Geiping, Jonas, McLeish, Sean, Jain, Neel, Kirchenbauer, John, Singh, Siddharth, Bartoldson, Brian R, Kailkhura, Bhavya, Bhatele, Abhinav, and Goldstein, Tom.* arXiv preprint arXiv:2502.05171 2025.
6.  [**Pretraining Language Models to Ponder in Continuous Space.**](https://arxiv.org/abs/2505.20674) *Zeng, Boyi, Song, Shixiang, Huang, Siyuan, Wang, Yixuan, Li, He, He, Ziwei, Wang, Xinbing, Li, Zhiyu, and Lin, Zhouhan.* arXiv preprint arXiv:2505.20674 2025.

##### Training-induced Recurrence
1.  [**Training large language models to reason in a continuous latent space.**](https://arxiv.org/abs/2412.06769) *Hao, Shibo, Sukhbaatar, Sainbayar, Su, DiJia, Li, Xian, Hu, Zhiting, Weston, Jason, and Tian, Yuandong.* arXiv preprint arXiv:2412.06769 2024.
2.  [**Codi: Compressing chain-of-thought into continuous space via self-distillation.**](https://arxiv.org/abs/2502.21074) *Shen, Zhenyi, Yan, Hanqi, Zhang, Linhai, Hu, Zhanghao, Du, Yali, and He, Yulan.* arXiv preprint arXiv:2502.21074 2025.
3.  [**Compressed chain of thought: Efficient reasoning through dense representations.**](https://arxiv.org/abs/2412.13171) *Cheng, Jeffrey, and Van Durme, Benjamin.* arXiv preprint arXiv:2412.13171 2024.
4.  [**Parallel Continuous Chain-of-Thought with Jacobi Iteration.**](https://arxiv.org/abs/2506.18582) *Wu, Haoyi, Teng, Zhihao, and Tu, Kewei.* arXiv preprint arXiv:2506.18582 2025.
5.  [**System-1.5 Reasoning: Traversal in Language and Latent Spaces with Dynamic Shortcuts.**](https://arxiv.org/abs/2505.18962) *Wang, Xiaoqiang, Wang, Suyuchen, Zhu, Yun, and Liu, Bang.* arXiv preprint arXiv:2505.18962 2025.
6.  [**Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning.**](https://arxiv.org/abs/2502.03275) *Su, DiJia, Zhu, Hanlin, Xu, Yingchen, Jiao, Jiantao, Tian, Yuandong, and Zheng, Qinqing.* arXiv preprint arXiv:2502.03275 2025.
7.  [**Lightthinker: Thinking step-by-step compression.**](https://arxiv.org/abs/2502.15589) *Zhang, Jintian, Zhu, Yuqi, Sun, Mengshu, Luo, Yujie, Qiao, Shuofei, Du, Lun, Zheng, Da, Chen, Huajun, and Zhang, Ningyu.* arXiv preprint arXiv:2502.15589 2025.
8.  [**Lettextbackslash textquoterights Think Dot by Dot: Hidden computation in transformer language models.**](https://openreview.net/forum?id=NikbrdtYvG) *Jacob Pfau, William Merrill, and Samuel R. Bowman.* First Conference on Language Modeling 2024.
9.  [**Think before you speak: Training Language Models With Pause Tokens.**](https://openreview.net/forum?id=ph04CRkPdC) *Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan.* The Twelfth International Conference on Learning Representations 2024.
10.  [**Guiding Language Model Reasoning with Planning Tokens.**](https://openreview.net/forum?id=wi9IffRhVM) *Xinyi Wang, Lucas Caccia, Oleksiy Ostapenko, Xingdi Yuan, William Yang Wang, and Alessandro Sordoni.* First Conference on Language Modeling 2024.
11.  [**Disentangling memory and reasoning ability in large language models.**](https://arxiv.org/abs/2411.13504) *Jin, Mingyu, Luo, Weidi, Cheng, Sitao, Wang, Xinyi, Hua, Wenyue, Tang, Ruixiang, Wang, William Yang, and Zhang, Yongfeng.* arXiv preprint arXiv:2411.13504 2024.

##### Training Strategies for Recurrent Reasoning
1.  [**On the inductive bias of stacking towards improving reasoning.**](#) *Saunshi, Nikunj, Karp, Stefani, Krishnan, Shankar, Miryoosefi, Sobhan, Jakkam Reddi, Sashank, and Kumar, Sanjiv.* Advances in Neural Information Processing Systems 2024.
2.  [**Reasoning with latent thoughts: On the power of looped transformers.**](https://arxiv.org/abs/2502.17416) *Saunshi, Nikunj, Dikkala, Nishanth, Li, Zhiyuan, Kumar, Sanjiv, and Reddi, Sashank J.* arXiv preprint arXiv:2502.17416 2025.
3.  [**From explicit cot to implicit cot: Learning to internalize cot step by step.**](https://arxiv.org/abs/2405.14838) *Deng, Yuntian, Choi, Yejin, and Shieber, Stuart.* arXiv preprint arXiv:2405.14838 2024.
4.  [**Training large language models to reason in a continuous latent space.**](https://arxiv.org/abs/2412.06769) *Hao, Shibo, Sukhbaatar, Sainbayar, Su, DiJia, Li, Xian, Hu, Zhiting, Weston, Jason, and Tian, Yuandong.* arXiv preprint arXiv:2412.06769 2024.
5.  [**Enhancing Auto-regressive Chain-of-Thought through Loop-Aligned Reasoning.**](https://arxiv.org/abs/2502.08482) *Yu, Qifan, He, Zhenyu, Li, Sijie, Zhou, Xun, Zhang, Jun, Xu, Jingjing, and He, Di.* arXiv preprint arXiv:2502.08482 2025.

##### Applications and Capabilities
1.  [**Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks.**](#) *Schwarzschild, Avi, Borgnia, Eitan, Gupta, Arjun, Huang, Furong, Vishkin, Uzi, Goldblum, Micah, and Goldstein, Tom.* Advances in Neural Information Processing Systems 2021.
2.  [**Looped transformers as programmable computers.**](#) *Giannou, Angeliki, Rajput, Shashank, Sohn, Jy-yong, Lee, Kangwook, Lee, Jason D, and Papailiopoulos, Dimitris.* International Conference on Machine Learning 2023.
3.  [**Simulation of graph algorithms with looped transformers.**](https://arxiv.org/abs/2402.01107) *De Luca, Artur Back, and Fountoulakis, Kimon.* arXiv preprint arXiv:2402.01107 2024.
4.  [**Guiding Language Model Reasoning with Planning Tokens.**](https://openreview.net/forum?id=wi9IffRhVM) *Xinyi Wang, Lucas Caccia, Oleksiy Ostapenko, Xingdi Yuan, William Yang Wang, and Alessandro Sordoni.* First Conference on Language Modeling 2024.
5.  [**Disentangling memory and reasoning ability in large language models.**](https://arxiv.org/abs/2411.13504) *Jin, Mingyu, Luo, Weidi, Cheng, Sitao, Wang, Xinyi, Hua, Wenyue, Tang, Ruixiang, Wang, William Yang, and Zhang, Yongfeng.* arXiv preprint arXiv:2411.13504 2024.
6.  [**Can looped transformers learn to implement multi-step gradient descent for in-context learning?.**](https://arxiv.org/abs/2410.08292) *Gatmiry, Khashayar, Saunshi, Nikunj, Reddi, Sashank J, Jegelka, Stefanie, and Kumar, Sanjiv.* arXiv preprint arXiv:2410.08292 2024.
7.  [**Bypassing the exponential dependency: Looped transformers efficiently learn in-context by multi-step gradient descent.**](https://arxiv.org/abs/2410.11268) *Chen, Bo, Li, Xiaoyu, Liang, Yingyu, Shi, Zhenmei, and Song, Zhao.* arXiv preprint arXiv:2410.11268 2024.

#### Temporal Hidden-state Methods

##### Hidden-state based methods
1.  [**Transformers are ssms: Generalized models and efficient algorithms through structured state space duality.**](https://arxiv.org/abs/2405.21060) *Dao, Tri, and Gu, Albert.* arXiv preprint arXiv:2405.21060 2024.
2.  [**Gated linear attention transformers with hardware-efficient training.**](https://arxiv.org/abs/2312.06635) *Yang, Songlin, Wang, Bailin, Shen, Yikang, Panda, Rameswar, and Kim, Yoon.* arXiv preprint arXiv:2312.06635 2023.
3.  [**Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence.**](https://arxiv.org/abs/2404.05892) *Peng, Bo, Goldstein, Daniel, Anthony, Quentin, Albalak, Alon, Alcaide, Eric, Biderman, Stella, Cheah, Eugene, Ferdinan, Teddy, Hou, Haowen, Kazienko, Przemyslaw, and others.* arXiv preprint arXiv:2404.05892 2024.
4.  [**Hgrn2: Gated linear rnns with state expansion.**](https://arxiv.org/abs/2404.07904) *Qin, Zhen, Yang, Songlin, Sun, Weixuan, Shen, Xuyang, Li, Dong, Sun, Weigao, and Zhong, Yiran.* arXiv preprint arXiv:2404.07904 2024.
5.  **Citation key not found:** `deltanet_citation`
6.  [**Parallelizing linear transformers with the delta rule over sequence length.**](https://arxiv.org/abs/2406.06484) *Yang, Songlin, Wang, Bailin, Zhang, Yu, Shen, Yikang, and Kim, Yoon.* arXiv preprint arXiv:2406.06484 2024.

##### Optimization-based State Evolution
1.  [**Transformers are rnns: Fast autoregressive transformers with linear attention.**](#) *Katharopoulos, Angelos, Vyas, Apoorv, Pappas, Nikolaos, and Fleuret, Franc cois.* International conference on machine learning 2020.
2.  [**Retentive network: A successor to transformer for large language models.**](https://arxiv.org/abs/2307.08621) *Sun, Yutao, Dong, Li, Huang, Shaohan, Ma, Shuming, Xia, Yuqing, Xue, Jilong, Wang, Jianyong, and Wei, Furu.* arXiv preprint arXiv:2307.08621 2023.
3.  [**Gated linear attention transformers with hardware-efficient training.**](https://arxiv.org/abs/2312.06635) *Yang, Songlin, Wang, Bailin, Shen, Yikang, Panda, Rameswar, and Kim, Yoon.* arXiv preprint arXiv:2312.06635 2023.
4.  [**Parallelizing linear transformers with the delta rule over sequence length.**](https://arxiv.org/abs/2406.06484) *Yang, Songlin, Wang, Bailin, Zhang, Yu, Shen, Yikang, and Kim, Yoon.* arXiv preprint arXiv:2406.06484 2024.
5.  [**Gated Delta Networks: Improving Mamba2 with Delta Rule.**](https://arxiv.org/abs/2412.06464) *Yang, Songlin, Kautz, Jan, and Hatamizadeh, Ali.* arXiv preprint arXiv:2412.06464 2024.
6.  [**Transformers are ssms: Generalized models and efficient algorithms through structured state space duality.**](https://arxiv.org/abs/2405.21060) *Dao, Tri, and Gu, Albert.* arXiv preprint arXiv:2405.21060 2024.
7.  [**Hgrn2: Gated linear rnns with state expansion.**](https://arxiv.org/abs/2404.07904) *Qin, Zhen, Yang, Songlin, Sun, Weixuan, Shen, Xuyang, Li, Dong, Sun, Weigao, and Zhong, Yiran.* arXiv preprint arXiv:2404.07904 2024.
8.  [**Eagle and finch: Rwkv with matrix-valued states and dynamic recurrence.**](https://arxiv.org/abs/2404.05892) *Peng, Bo, Goldstein, Daniel, Anthony, Quentin, Albalak, Alon, Alcaide, Eric, Biderman, Stella, Cheah, Eugene, Ferdinan, Teddy, Hou, Haowen, Kazienko, Przemyslaw, and others.* arXiv preprint arXiv:2404.05892 2024.
9.  [**Learning to (learn at test time): Rnns with expressive hidden states.**](https://arxiv.org/abs/2407.04620) *Sun, Yu, Li, Xinhao, Dalal, Karan, Xu, Jiarui, Vikram, Arjun, Zhang, Genghan, Dubois, Yann, Chen, Xinlei, Wang, Xiaolong, Koyejo, Sanmi, and others.* arXiv preprint arXiv:2407.04620 2024.
10.  [**Titans: Learning to memorize at test time.**](https://arxiv.org/abs/2501.00663) *Behrouz, Ali, Zhong, Peilin, and Mirrokni, Vahab.* arXiv preprint arXiv:2501.00663 2024.
11.  [**Lattice: Learning to efficiently compress the memory.**](https://arxiv.org/abs/2504.05646) *Karami, Mahdi, and Mirrokni, Vahab.* arXiv preprint arXiv:2504.05646 2025.
12.  [**Ittextquotesingle s All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization.**](https://arxiv.org/abs/2504.13173) *Behrouz, Ali, Razaviyayn, Meisam, Zhong, Peilin, and Mirrokni, Vahab.* arXiv preprint arXiv:2504.13173 2025.
13.  [**Atlas: Learning to optimally memorize the context at test time.**](https://arxiv.org/abs/2505.23735) *Behrouz, Ali, Li, Zeman, Kacham, Praneeth, Daliri, Majid, Deng, Yuan, Zhong, Peilin, Razaviyayn, Meisam, and Mirrokni, Vahab.* arXiv preprint arXiv:2505.23735 2025.
14.  [**Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration.**](https://arxiv.org/abs/2505.24688) *Zhu, Qinglin, Zhao, Runcong, Yan, Hanqi, He, Yulan, Chen, Yudong, and Gui, Lin.* arXiv preprint arXiv:2505.24688 2025.

##### Training-induced Hidden-State Conversion
1.  [**Transformers to ssms: Distilling quadratic knowledge to subquadratic models.**](#) *Bick, Aviv, Li, Kevin, Xing, Eric, Kolter, J Zico, and Gu, Albert.* Advances in Neural Information Processing Systems 2024.
2.  [**Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing.**](https://arxiv.org/abs/2502.14458) *Aviv Bick, Tobias Katsch, Nimit Sohoni, Arjun Desai, and Albert Gu.* arXiv preprint 2025.
3.  [**Linearizing large language models.**](https://arxiv.org/abs/2405.06640) *Mercat, Jean, Vasiljevic, Igor, Keh, Sedrick, Arora, Kushal, Dave, Achal, Gaidon, Adrien, and Kollar, Thomas.* arXiv preprint arXiv:2405.06640 2024.
4.  [**LoLCATs: On Low-Rank Linearizing of Large Language Models.**](https://arxiv.org/abs/2410.10254) *Zhang, Michael, Arora, Simran, Chalamala, Rahul, Wu, Alan, Spector, Benjamin, Singhal, Aaryan, Ramesh, Krithik, and R'e, Christopher.* arXiv preprint arXiv:2410.10254 2024.
5.  [**Liger: Linearizing Large Language Models to Gated Recurrent Structures.**](https://arxiv.org/abs/2503.01496) *Lan, Disen, Sun, Weigao, Hu, Jiaxi, Du, Jusen, and Cheng, Yu.* arXiv preprint arXiv:2503.01496 2025.

### Mechanistic Interpretability

#### Do Layer Stacks Reflect Latent CoT?
1.  [**Openai o1 system card.**](https://arxiv.org/abs/2412.16720) *Jaech, Aaron, Kalai, Adam, Lerer, Adam, Richardson, Adam, El-Kishky, Ahmed, Low, Aiden, Helyar, Alec, Madry, Aleksander, Beutel, Alex, Carney, Alex, and others.* arXiv preprint arXiv:2412.16720 2024.
2.  [**Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.**](https://arxiv.org/abs/2501.12948) *Guo, Daya, Yang, Dejian, Zhang, Haowei, Song, Junxiao, Zhang, Ruoyu, Xu, Runxin, Zhu, Qihao, Ma, Shirong, Wang, Peiyi, Bi, Xiao, and others.* arXiv preprint arXiv:2501.12948 2025.
3.  [**Do LLMs Really Think Step-by-step In Implicit Reasoning?.**](https://arxiv.org/abs/2411.15862) *Yu, Yijiong.* arXiv preprint arXiv:2411.15862 2024.
4.  [**How Do LLMs Perform Two-Hop Reasoning in Context?.**](https://arxiv.org/abs/2502.13913) *Guo, Tianyu, Zhu, Hanlin, Zhang, Ruiqi, Jiao, Jiantao, Mei, Song, Jordan, Michael I, and Russell, Stuart.* arXiv preprint arXiv:2502.13913 2025.
5.  [**Reasoning with latent thoughts: On the power of looped transformers.**](https://arxiv.org/abs/2502.17416) *Saunshi, Nikunj, Dikkala, Nishanth, Li, Zhiyuan, Kumar, Sanjiv, and Reddi, Sashank J.* arXiv preprint arXiv:2502.17416 2025.
6.  [**A little depth goes a long way: The expressive power of log-depth transformers.**](https://arxiv.org/abs/2503.03961) *Merrill, William, and Sabharwal, Ashish.* arXiv preprint arXiv:2503.03961 2025.
7.  [**Towards a mechanistic interpretation of multi-step reasoning capabilities of language models.**](https://arxiv.org/abs/2310.14491) *Hou, Yifan, Li, Jiaoda, Fei, Yu, Stolfo, Alessandro, Zhou, Wangchunshu, Zeng, Guangtao, Bosselut, Antoine, and Sachan, Mrinmaya.* arXiv preprint arXiv:2310.14491 2023.
8.  [**Iteration head: A mechanistic study of chain-of-thought.**](#) *Cabannes, Vivien, Arnal, Charles, Bouaziz, Wassim, Yang, Xingyu, Charton, Francois, and Kempe, Julia.* Advances in Neural Information Processing Systems 2024.
9.  [**Towards understanding how transformer perform multi-step reasoning with matching operation.**](#) *Wang, Zhiwei, Wang, Yunji, Zhang, Zhongwang, Zhou, Zhangchen, Jin, Hui, Hu, Tianyang, Sun, Jiacheng, Li, Zhenguo, Zhang, Yaoyu, and Xu, Zhi-Qin John.* arXiv e-prints 2024.
10.  [**Back attention: Understanding and enhancing multi-hop reasoning in large language models.**](https://arxiv.org/abs/2502.10835) *Yu, Zeping, Belinkov, Yonatan, and Ananiadou, Sophia.* arXiv preprint arXiv:2502.10835 2025.

#### Mechanisms of Latent CoT in Layer Representation
1.  [**Investigating layer importance in large language models.**](https://arxiv.org/abs/2409.14381) *Zhang, Yang, Dong, Yanfei, and Kawaguchi, Kenji.* arXiv preprint arXiv:2409.14381 2024.
2.  [**The Unreasonable Ineffectiveness of the Deeper Layers.**](https://arxiv.org/abs/2403.17887) *Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A. Roberts.* arXiv preprint arXiv:2403.17887 2024.
3.  [**Understanding Layer Significance in LLM Alignment.**](https://arxiv.org/abs/2410.17875) *Shi, Guangyuan, Lu, Zexin, Dong, Xiaoyu, Zhang, Wenlong, Zhang, Xuanyu, Feng, Yujie, and Wu, Xiao-Ming.* arXiv preprint arXiv:2410.17875 2024.
4.  [**micse: Mutual information contrastive learning for low-shot sentence embeddings.**](https://arxiv.org/abs/2211.04928) *Klein, Tassilo, and Nabi, Moin.* arXiv preprint arXiv:2211.04928 2022.
5.  [**Transformer feed-forward layers are key-value memories.**](https://arxiv.org/abs/2012.14913) *Geva, Mor, Schuster, Roei, Berant, Jonathan, and Levy, Omer.* arXiv preprint arXiv:2012.14913 2020.
6.  [**Unveiling induction heads: Provable training dynamics and feature learning in transformers.**](https://arxiv.org/abs/2409.10559) *Chen, Siyu, Sheen, Heejune, Wang, Tianhao, and Yang, Zhuoran.* arXiv preprint arXiv:2409.10559 2024.
7.  [**Loss landscape geometry reveals stagewise development of transformers.**](#) *Wang, George, Farrugia-Roberts, Matthew, Hoogland, Jesse, Carroll, Liam, Wei, Susan, and Murfet, Daniel.* High-dimensional Learning Dynamics 2024: The Emergence of Structure and Reasoning 2024.
8.  [**Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting.**](#) *Li, Shiyang, Jin, Xiaoyong, Xuan, Yao, Zhou, Xiyou, Chen, Wenhu, Wang, Yu-Xiang, and Yan, Xifeng.* Advances in neural information processing systems 2019.
9.  [**Unpacking Robustness in Inflectional Languages: Adversarial Evaluation and Mechanistic Insights.**](https://arxiv.org/abs/2505.07856) *Walkowiak, Pawe'L, Klonowski, Marek, Oleksy, Marcin, and Janz, Arkadiusz.* arXiv preprint arXiv:2505.07856 2025.
10.  [**Do large language models latently perform multi-hop reasoning?.**](https://arxiv.org/abs/2402.16837) *Yang, Sohee, Gribovskaya, Elena, Kassner, Nora, Geva, Mor, and Riedel, Sebastian.* arXiv preprint arXiv:2402.16837 2024.
11.  [**Layer by Layer: Uncovering Hidden Representations in Language Models.**](https://arxiv.org/abs/2502.02013) *Skean, Oscar, Arefin, Md Rifat, Zhao, Dan, Patel, Niket, Naghiyev, Jalal, LeCun, Yann, and Shwartz-Ziv, Ravid.* arXiv preprint arXiv:2502.02013 2025.
12.  [**Distributional reasoning in llms: Parallel reasoning processes in multi-hop reasoning.**](https://arxiv.org/abs/2406.13858) *Shalev, Yuval, Feder, Amir, and Goldstein, Ariel.* arXiv preprint arXiv:2406.13858 2024.
13.  [**Hopping too late: Exploring the limitations of large language models on multi-hop queries.**](https://arxiv.org/abs/2406.12775) *Biran, Eden, Gottesman, Daniela, Yang, Sohee, Geva, Mor, and Globerson, Amir.* arXiv preprint arXiv:2406.12775 2024.
14.  [**Interpretability in the wild: a circuit for indirect object identification in gpt-2 small.**](https://arxiv.org/abs/2211.00593) *Wang, Kevin, Variengien, Alexandre, Conmy, Arthur, Shlegeris, Buck, and Steinhardt, Jacob.* arXiv preprint arXiv:2211.00593 2022.
15.  [**How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model.**](#) *Hanna, Michael, Liu, Ollie, and Variengien, Alexandre.* Advances in Neural Information Processing Systems 2023.
16.  [**Does representation matter? exploring intermediate layers in large language models.**](https://arxiv.org/abs/2412.09563) *Skean, Oscar, Arefin, Md Rifat, LeCun, Yann, and Shwartz-Ziv, Ravid.* arXiv preprint arXiv:2412.09563 2024.
17.  [**Unifying and Verifying Mechanistic Interpretations: A Case Study with Group Operations.**](https://arxiv.org/abs/2410.07476) *Wu, Wilson, Jaburi, Louis, Drori, Jacob, and Gross, Jason.* arXiv preprint arXiv:2410.07476 2024.
18.  [**Emergent Abilities in Large Language Models: A Survey.**](https://arxiv.org/abs/2503.05788) *Berti, Leonardo, Giorgi, Flavio, and Kasneci, Gjergji.* arXiv preprint arXiv:2503.05788 2025.
19.  [**An explainable transformer circuit for compositional generalization.**](https://arxiv.org/abs/2502.15801) *Tang, Cheng, Lake, Brenden, and Jazayeri, Mehrdad.* arXiv preprint arXiv:2502.15801 2025.
20.  [**The Representation and Recall of Interwoven Structured Knowledge in LLMs: A Geometric and Layered Analysis.**](https://arxiv.org/abs/2502.10871) *Lei, Ge, and Cooper, Samuel J.* arXiv preprint arXiv:2502.10871 2025.
21.  [**Unveiling Factual Recall Behaviors of Large Language Models through Knowledge Neurons.**](https://arxiv.org/abs/2408.03247) *Wang, Yifei, Chen, Yuheng, Wen, Wanting, Sheng, Yu, Li, Linjing, and Zeng, Daniel Dajun.* arXiv preprint arXiv:2408.03247 2024.
22.  [**Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.**](https://arxiv.org/abs/2502.05171) *Geiping, Jonas, McLeish, Sean, Jain, Neel, Kirchenbauer, John, Singh, Siddharth, Bartoldson, Brian R, Kailkhura, Bhavya, Bhatele, Abhinav, and Goldstein, Tom.* arXiv preprint arXiv:2502.05171 2025.
23.  [**How to think step-by-step: A mechanistic understanding of chain-of-thought reasoning.**](https://arxiv.org/abs/2402.18312) *Dutta, Subhabrata, Singh, Joykirat, Chakrabarti, Soumen, and Chakraborty, Tanmoy.* arXiv preprint arXiv:2402.18312 2024.
24.  [**Why lift so heavy? slimming large language models by cutting off the layers.**](https://arxiv.org/abs/2402.11700) *Yuan, Shuzhou, Nie, Ercong, Ma, Bolei, and F"arber, Michael.* arXiv preprint arXiv:2402.11700 2024.
25.  [**Do Language Models Use Their Depth Efficiently?.**](https://arxiv.org/abs/2505.13898) *Csord'as, R'obert, Manning, Christopher D, and Potts, Christopher.* arXiv preprint arXiv:2505.13898 2025.
26.  [**Void in Language Models.**](https://arxiv.org/abs/2505.14467) *Shemiranifar, Mani.* arXiv preprint arXiv:2505.14467 2025.
27.  [**The Curse of Depth in Large Language Models.**](https://arxiv.org/abs/2502.05795) *Sun, Wenfang, Song, Xinyuan, Li, Pengxiang, Yin, Lu, Zheng, Yefeng, and Liu, Shiwei.* arXiv preprint arXiv:2502.05795 2025.
28.  [**Inheritune: Training Smaller Yet More Attentive Language Models.**](https://arxiv.org/abs/2404.08634) *Sanyal, Sunny, Shwartz-Ziv, Ravid, Dimakis, Alex, and Sanghavi, Sujay.* arXiv preprint arXiv:2404.08634 2024.
29.  [**A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis.**](https://arxiv.org/abs/2305.15054) *Stolfo, Alessandro, Belinkov, Yonatan, and Sachan, Mrinmaya.* arXiv preprint arXiv:2305.15054 2023.
30.  [**Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization.**](https://arxiv.org/abs/2405.15071) *Wang, Boshi, Yue, Xiang, Su, Yu, and Sun, Huan.* arXiv preprint arXiv:2405.15071 2024.
31.  [**Back attention: Understanding and enhancing multi-hop reasoning in large language models.**](https://arxiv.org/abs/2502.10835) *Yu, Zeping, Belinkov, Yonatan, and Ananiadou, Sophia.* arXiv preprint arXiv:2502.10835 2025.

#### Turing Completeness of Layer-Based Latent CoT
1.  [**Attention is all you need.**](#) *Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones, Llion, Gomez, Aidan N, Kaiser, Lukasz, and Polosukhin, Illia.* Advances in neural information processing systems 2017.
2.  [**An outsidertextquotesingle s view of neural nets.**](#) *Jordan, Michael I..* Cognitive Science 1986.
3.  **Citation key not found:** `Elman1990FindingStructure`
4.  [**On the computational power of neural nets.**](#) *Siegelmann, Hava T., and Sontag, Eduardo D..* Journal of Computer and System Sciences 1995.
5.  [**Long Short-Term Memory.**](#) *Hochreiter, Sepp, and Schmidhuber, J"urgen.* Neural Computation 1997.
6.  [**Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.**](#) *Cho, Kyunghyun, Van Merri"enboer, Bart, Gulcehre, Caglar, Bahdanau, Dzmitry, Bougares, Fethi, Schwenk, Holger, and Bengio, Yoshua.* Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP) 2014.
7.  [**On the turing completeness of modern neural network architectures.**](https://arxiv.org/abs/1901.03429) *P'erez, Jorge, Marinkovi' c, Javier, and Barcel'o, Pablo.* arXiv preprint arXiv:1901.03429 2019.
8.  [**Constant Bit-size Transformers Are Turing Complete.**](https://arxiv.org/abs/2506.12027) *Li, Qian, and Wang, Yuyi.* arXiv preprint arXiv:2506.12027 2025.
9.  [**Ask, and it shall be given: On the Turing completeness of prompting.**](https://arxiv.org/abs/2411.01992) *Qiu, Ruizhong, Xu, Zhe, Bao, Wenxuan, and Tong, Hanghang.* arXiv preprint arXiv:2411.01992 2024.
10.  [**Chain of thought empowers transformers to solve inherently serial problems.**](https://arxiv.org/abs/2402.12875) *Li, Zhiyuan, Liu, Hong, Zhou, Denny, and Ma, Tengyu.* arXiv preprint arXiv:2402.12875 2024.
11.  [**Recurrent memory transformer.**](#) *Bulatov, Aydar, Kuratov, Yury, and Burtsev, Mikhail.* Advances in Neural Information Processing Systems 2022.
12.  [**Investigating Recurrent Transformers with Dynamic Halt.**](https://arxiv.org/abs/2402.00976) *Chowdhury, Jishnu Ray, and Caragea, Cornelia.* arXiv preprint arXiv:2402.00976 2024.
13.  [**Quiet-star: Language models can teach themselves to think before speaking.**](https://arxiv.org/abs/2403.09629) *Zelikman, Eric, Harik, Georges, Shao, Yijia, Jayasiri, Varuna, Haber, Nick, and Goodman, Noah D.* arXiv preprint arXiv:2403.09629 2024.
14.  [**Reinforcement Pre-Training.**](https://arxiv.org/abs/2506.08007) *Qingxiu Dong, Li Dong, Yao Tang, Tianzhu Ye, Yutao Sun, Zhifang Sui, and Furu Wei.* arXiv preprint arXiv:2506.08007 2025.
15.  [**On limitations of the transformer architecture.**](#) *Peng, Binghui, Narayanan, Srini, and Papadimitriou, Christos.* First Conference on Language Modeling 2024.
16.  [**Looped transformers as programmable computers.**](#) *Giannou, Angeliki, Rajput, Shashank, Sohn, Jy-yong, Lee, Kangwook, Lee, Jason D, and Papailiopoulos, Dimitris.* International Conference on Machine Learning 2023.

### Towards Infinite-depth Reasoning

#### Spatial Infinite Reasoning: Text Diffusion Models

##### Masked Diffusion Models
1.  [**Structured denoising diffusion models in discrete state-spaces.**](#) *Austin, Jacob, Johnson, Daniel D, Ho, Jonathan, Tarlow, Daniel, and Van Den Berg, Rianne.* Advances in neural information processing systems 2021.
2.  [**Discrete diffusion modeling by estimating the ratios of the data distribution.**](#) *Lou, Aaron, Meng, Chenlin, and Ermon, Stefano.* Proceedings of the 41st International Conference on Machine Learning 2024.
3.  [**Your absorbing discrete diffusion secretly models the conditional distributions of clean data.**](https://arxiv.org/abs/2406.03736) *Ou, Jingyang, Nie, Shen, Xue, Kaiwen, Zhu, Fengqi, Sun, Jiacheng, Li, Zhenguo, and Li, Chongxuan.* arXiv preprint arXiv:2406.03736 2024.
4.  [**Simplified and generalized masked diffusion for discrete data.**](#) *Shi, Jiaxin, Han, Kehang, Wang, Zhe, Doucet, Arnaud, and Titsias, Michalis.* Advances in neural information processing systems 2024.
5.  [**Simple and effective masked diffusion language models.**](#) *Sahoo, Subham, Arriola, Marianne, Schiff, Yair, Gokaslan, Aaron, Marroquin, Edgar, Chiu, Justin, Rush, Alexander, and Kuleshov, Volodymyr.* Advances in Neural Information Processing Systems 2024.
6.  [**Scaling up Masked Diffusion Models on Text.**](https://arxiv.org/abs/2410.18514) *Nie, Shen, Zhu, Fengqi, Du, Chao, Pang, Tianyu, Liu, Qian, Zeng, Guangtao, Lin, Min, and Li, Chongxuan.* arXiv preprint arXiv:2410.18514 2024.
7.  [**Learning Iterative Reasoning through Energy Diffusion.**](#) *Du, Yilun, Mao, Jiayuan, and Tenenbaum, Joshua B..* International Conference on Machine Learning (ICML) 2024.
8.  [**TESS 2: A Large-Scale Generalist Diffusion Language Model.**](https://arxiv.org/abs/2502.13917) *Tae, Jaesung, Ivison, Hamish, Kumar, Sachin, and Cohan, Arman.* arXiv preprint arXiv:2502.13917 2025.
9.  [**Large Language Diffusion Models.**](#) *Nie, Shen, Zhu, Fengqi, You, Zebin, Zhang, Xiaolu, Ou, Jingyang, Hu, Jun, ZHOU, JUN, Lin, Yankai, Wen, Ji-Rong, and Li, Chongxuan.* ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy 2025.

##### Chain-of-Thought Diffusion Models
1.  [**Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models.**](https://arxiv.org/abs/2402.07754) *Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Xin Jiang, Zhenguo Li, Wei Bi, and Lingpeng Kong.* 2024.
2.  [**Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning.**](https://arxiv.org/abs/2410.14157) *Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong.* 2025.

##### Hybrid Diffusion and Autoregressive Architectures
1.  [**Large Language Models to Diffusion Finetuning.**](https://arxiv.org/abs/2501.15781) *Cetin, Edoardo, Zhao, Tianyu, and Tang, Yujin.* arXiv preprint arXiv:2501.15781 2025.
2.  [**Dream 7B.**](https://hkunlp.github.io/blog/2025/dream) *Ye, Jiacheng, Xie, Zhihui, Zheng, Lin, Gao, Jiahui, Wu, Zirui, Jiang, Xin, Li, Zhenguo, and Kong, Lingpeng.* 2025.
3.  [**Scaling Diffusion Language Models via Adaptation from Autoregressive Models.**](https://openreview.net/forum?id=j1tSLYKwg8) *Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, Hao Peng, and Lingpeng Kong.* The Thirteenth International Conference on Learning Representations 2025.
4.  [**Mercury: Ultra-Fast Language Models Based on Diffusion.**](#) *Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, and Volodymyr Kuleshov.* 2025.
5.  [**Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities.**](#) *Gemini, Team.* 2025.
6.  [**Diffusion of Thought: Chain-of-Thought Reasoning in Diffusion Language Models.**](#) *Ye, Jiacheng, Gong, Shansan, Chen, Liheng, Zheng, Lin, Gao, Jiahui, Shi, Han, Wu, Chuan, Jiang, Xin, Li, Zhenguo, Bi, Wei, and others.* The Thirty-eighth Annual Conference on Neural Information Processing Systems 2025.
7.  [**Learning Iterative Reasoning through Energy Diffusion.**](#) *Du, Yilun, Mao, Jiayuan, and Tenenbaum, Joshua B..* International Conference on Machine Learning (ICML) 2024.
8.  [**Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models.**](https://arxiv.org/abs/2402.07754) *Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Xin Jiang, Zhenguo Li, Wei Bi, and Lingpeng Kong.* 2024.
9.  [**Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning.**](https://arxiv.org/abs/2410.14157) *Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong.* 2025.
10.  [**Scaling up Masked Diffusion Models on Text.**](https://arxiv.org/abs/2410.18514) *Nie, Shen, Zhu, Fengqi, Du, Chao, Pang, Tianyu, Liu, Qian, Zeng, Guangtao, Lin, Min, and Li, Chongxuan.* arXiv preprint arXiv:2410.18514 2024.
11.  [**Large Language Diffusion Models.**](#) *Nie, Shen, Zhu, Fengqi, You, Zebin, Zhang, Xiaolu, Ou, Jingyang, Hu, Jun, ZHOU, JUN, Lin, Yankai, Wen, Ji-Rong, and Li, Chongxuan.* ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy 2025.
12.  [**Simple and effective masked diffusion language models.**](#) *Sahoo, Subham, Arriola, Marianne, Schiff, Yair, Gokaslan, Aaron, Marroquin, Edgar, Chiu, Justin, Rush, Alexander, and Kuleshov, Volodymyr.* Advances in Neural Information Processing Systems 2024.
13.  [**TESS 2: A Large-Scale Generalist Diffusion Language Model.**](https://arxiv.org/abs/2502.13917) *Tae, Jaesung, Ivison, Hamish, Kumar, Sachin, and Cohan, Arman.* arXiv preprint arXiv:2502.13917 2025.


#### Towards an 'Infinitely Long' Optimiser Network
1.  [**Leave no context behind: Efficient infinite context transformers with infini-attention.**](https://arxiv.org/abs/2404.07143) *Munkhdalai, Tsendsuren, Faruqui, Manaal, and Gopal, Siddharth.* arXiv preprint arXiv:2404.07143 2024.
2.  [**Learning to (learn at test time): Rnns with expressive hidden states.**](https://arxiv.org/abs/2407.04620) *Sun, Yu, Li, Xinhao, Dalal, Karan, Xu, Jiarui, Vikram, Arjun, Zhang, Genghan, Dubois, Yann, Chen, Xinlei, Wang, Xiaolong, Koyejo, Sanmi, and others.* arXiv preprint arXiv:2407.04620 2024.
3.  [**Titans: Learning to memorize at test time.**](https://arxiv.org/abs/2501.00663) *Behrouz, Ali, Zhong, Peilin, and Mirrokni, Vahab.* arXiv preprint arXiv:2501.00663 2024.
4.  [**Atlas: Learning to optimally memorize the context at test time.**](https://arxiv.org/abs/2505.23735) *Behrouz, Ali, Li, Zeman, Kacham, Praneeth, Daliri, Majid, Deng, Yuan, Zhong, Peilin, Razaviyayn, Meisam, and Mirrokni, Vahab.* arXiv preprint arXiv:2505.23735 2025.

#### Implicit Fixed‑Point RNNs
1.  [**Implicit Language Models are RNNs: Balancing Parallelization and Expressivity.**](https://arxiv.org/abs/2502.07827) *Sch"one, Mark, Rahmani, Babak, Kremer, Heiner, Falck, Fabian, Ballani, Hitesh, and Gladrow, Jannes.* arXiv preprint arXiv:2502.07827 2025.

#### Discussion
1.  [**A survey of diffusion models in natural language processing.**](https://arxiv.org/abs/2305.14671) *Zou, Hao, Kim, Zae Myung, and Kang, Dongyeop.* arXiv preprint arXiv:2305.14671 2023.
2.  [**Large Language Diffusion Models.**](#) *Nie, Shen, Zhu, Fengqi, You, Zebin, Zhang, Xiaolu, Ou, Jingyang, Hu, Jun, ZHOU, JUN, Lin, Yankai, Wen, Ji-Rong, and Li, Chongxuan.* ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy 2025.
3.  [**Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis.**](https://arxiv.org/abs/2412.04431) *Han, Jian, Liu, Jinlai, Jiang, Yi, Yan, Bin, Zhang, Yuqi, Yuan, Zehuan, Peng, Bingyue, and Liu, Xiaobing.* arXiv preprint arXiv:2412.04431 2024.
