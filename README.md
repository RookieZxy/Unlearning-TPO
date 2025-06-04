# Unlearning-TPO
This is the official repository for "[Not All Tokens Are Meant to Be Forgotten](https://arxiv.org/pdf/2506.03142)" by [Xiangyu Zhou](https://www.linkedin.com/in/xiangyu-zhou-71086321a/), [Yao Qiang](https://qiangyao1988.github.io/), [Saleh Zare Zade](https://scholar.google.com/citations?user=O3X_iagAAAAJ&hl=en&oi=ao), [Douglas Zytko](https://dougzytko.com/), [Prashant Khanduri](https://sites.google.com/view/khanduri-prashant/), [Dongxiao Zhu](https://dongxiaozhu.github.io/)

<table align="center">
  <tr>
    <td align="center"> 
      <img src="images/Idea Figure.png" alt="Teaser" style="width: 900px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Illustration of the proposed TIF framework.</em>
    </td>
  </tr>
</table>

## Abstract
Large Language Models (LLMs), pre-trained on massive text corpora, exhibit remarkable human-level language understanding, reasoning, and decision-making abilities. However, they tend to memorize unwanted information, such as private or copyrighted content, raising significant privacy and legal concerns. Unlearning has emerged as a promising solution, but existing methods face a significant challenge of over-forgetting. This issue arises because they indiscriminately suppress the generation of all the tokens in forget samples, leading to a substantial loss of model utility. To overcome this challenge, we introduce the Targeted Information Forgetting (TIF) framework, which consists of (1) a flexible targeted information identifier designed to differentiate between unwanted words (UW) and general words (GW) in the forget samples, and (2) a novel Targeted Preference Optimization approach that leverages Logit Preference Loss to unlearn unwanted information associated with UW and Preservation Loss to retain general information in GW, effectively improving the unlearning process while mitigating utility degradation. Extensive experiments on the TOFU and MUSE benchmarks demonstrate that the proposed TIF framework enhances unlearning effectiveness while preserving model utility and achieving state-of-the-art results.

## Getting Started
- [TPO on TOFU](TOFU)
- [TPO on MUSE](MUSE)

## Cite This Work
```bash



