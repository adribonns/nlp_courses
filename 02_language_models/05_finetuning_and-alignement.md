# 🎯 5. Fine-Tuning and Alignment of Large Language Models

After large-scale pretraining, LLMs possess vast linguistic and factual knowledge — but they are not yet *helpful, honest, or safe*.  
Fine-tuning and alignment techniques bridge that gap between *raw language modeling* and *human-compatible intelligence*.

---

## 5.1 The Base Model: Next-Token Prediction

The **base model** (foundation model) is pretrained on large-scale text data using the **Causal Language Modeling (CLM)** objective:

$$
\mathcal{L}_{\text{CLM}} = - \sum_{t} \log P_\theta(w_t | w_{<t})
$$

This objective teaches the model to predict the next token given previous ones — enabling fluent text generation.

However, such models:
- may produce toxic or biased content,
- can be verbose or unhelpful,
- don’t necessarily follow user instructions.

Hence, *alignment* is needed.

---

## 5.2 Stages of Model Adaptation

| Stage | Description | Example |
|:--|:--|:--|
| **Pretraining** | Predict next token on massive text corpus | GPT, LLaMA base |
| **Supervised Fine-Tuning (SFT)** | Train on curated (instruction → response) pairs | InstructGPT |
| **Alignment (RLHF / DPO)** | Optimize for human preference / safety | ChatGPT, Claude |

---

## 5.3 Supervised Fine-Tuning (SFT)

The first step beyond pretraining is **Supervised Fine-Tuning**.

The model is trained on datasets of human-written instruction–response pairs:

$$
\mathcal{L}_{\text{SFT}} = - \sum_{(x, y)} \log P_\theta(y | x)
$$

This teaches the model to **follow explicit user instructions**, improving usefulness and coherence.

### Example dataset
User: Summarize this paragraph.
Assistant: [gold human answer


### Output
The resulting model is an **instruction-tuned model** (e.g., FLAN-T5, Alpaca, LLaMA-2-Chat).

---

## 5.4 Alignment and Reinforcement Learning from Human Feedback (RLHF)

Even after SFT, models can:
- hallucinate,
- produce unsafe outputs,
- ignore nuanced preferences.

RLHF introduces a *reward signal* derived from **human feedback**.

### RLHF pipeline

1. **SFT Model** – provides initial responses.  
2. **Reward Model (RM)** – trained on human preference data.  
   - Humans rank multiple responses for the same prompt.
   - The RM learns to predict a scalar reward $r$ for a response.
3. **Policy Optimization** – the base model is fine-tuned to maximize expected reward.

---

### 5.4.1 The Reinforcement Learning Framework

We model text generation as a **policy** $\pi_\theta(y|x)$ that outputs a response $y$ given input $x$.

**Goal:**

$$
\max_\theta \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} [r(y)]
$$

The reward $r(y)$ reflects human preference quality.

---

### 5.4.2 PPO (Proximal Policy Optimization)

Used by OpenAI for InstructGPT and ChatGPT.

PPO optimizes the policy while constraining it not to deviate too far from the reference model (to avoid language drift).

**Objective:**

$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_t \Big[ 
\min \big(
r_t(\theta) \hat{A}_t,\ 
\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t
\big)
\Big]
$$

where
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$  
- $\hat{A}_t$ = advantage estimate (reward - baseline)

This stabilizes updates and prevents large jumps in policy behavior.

---

### 5.4.3 DPO (Direct Preference Optimization)

DPO (Rafailov et al., 2023) simplifies RLHF by removing the reinforcement loop.

Instead of sampling and updating via PPO, DPO **directly optimizes** the policy to prefer chosen responses over rejected ones.

Given pairs $(x, y^+, y^-)$ where $y^+$ is preferred over $y^-$:

$$
\mathcal{L}_{\text{DPO}} = - \log \sigma\left( 
\beta \left( \log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x) 
- \log \pi_{\text{ref}}(y^+|x) + \log \pi_{\text{ref}}(y^-|x)
\right) \right)
$$

Advantages:
- No reward model needed.
- Fully differentiable, stable, and efficient.
- Easier to train on large datasets.

---

### 5.4.4 PPO vs DPO Summary

| Aspect | PPO (RLHF) | DPO |
|:--|:--|:--|
| Uses reward model | ✅ Yes | ❌ No |
| Requires policy rollouts | ✅ Yes | ❌ No |
| Optimization style | Reinforcement Learning | Direct likelihood |
| Stability | Moderate | High |
| Compute cost | High | Lower |
| Interpretability | Moderate | Clear |

---

## 5.5 Adapters and Parameter-Efficient Fine-Tuning (PEFT)

Training the entire LLM is costly.  
PEFT methods tune only a small subset of parameters while keeping the backbone frozen.

### 5.5.1 LoRA (Low-Rank Adaptation)

Replace weight matrix $W$ by:

$$
W' = W + BA
$$

where $A, B$ are low-rank matrices ($r \ll d$).

Only $A$ and $B$ are trained → huge savings in memory and compute.

**Use cases:** domain adaptation, instruction-tuning, style transfer.

---

### 5.5.2 Adapters

Small neural modules inserted between Transformer layers, trained for new tasks while keeping base weights fixed.

**Example:**
Input → [Transformer Layer] → Adapter → Output


Adapters are parameter-efficient and allow *multi-task modularity*.

---

## 5.6 Alignment Beyond Reward Learning

Other alignment approaches include:
- **RL-free preference learning (DPO, IPO, ORPO)**  
- **Constitutional AI (Anthropic)**: AI follows written ethical principles instead of human labels.  
- **Self-alignment**: model critiques and refines its own responses (Self-Refine, Reflexion).  
- **Multi-turn feedback**: fine-tuning with dialogue history.

---

## 5.7 Evaluation and Monitoring

Key metrics during fine-tuning and alignment:
| Category | Metric | Description |
|:--|:--|:--|
| **Utility** | Win rate, helpfulness score | Alignment with user intent |
| **Factuality** | TruthfulQA, MMLU | Accuracy of knowledge |
| **Safety** | Toxicity, refusal rate | Ethical behavior |
| **Diversity** | Perplexity, repetition | Output variety |

---

## 5.8 Conceptual Diagram (Textual)

[Pretraining]
↓ (next token prediction)
[Supervised Fine-Tuning]
↓ (instruction datasets)
[Preference Optimization]
↓ (human feedback, PPO/DPO)
[Aligned Chat Model]


---

## 🧾 References

- Christiano et al. (2017). *Deep Reinforcement Learning from Human Preferences.*  
- Stiennon et al. (2020). *Learning to Summarize with Human Feedback.*  
- Ouyang et al. (2022). *Training Language Models to Follow Instructions (InstructGPT).*  
- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.*  
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*  
- Anthropic (2023). *Constitutional AI: Harmlessness from AI Feedback.*

---

## ✅ Summary

> Fine-tuning and alignment transform a raw pretrained model into a useful assistant.  
> Starting from next-token prediction, models are refined via supervised instruction data and preference optimization (RLHF, DPO).  
> Efficient tuning methods like LoRA and adapters enable continual adaptation, while new alignment methods improve safety and cooperation with human intent.
]
