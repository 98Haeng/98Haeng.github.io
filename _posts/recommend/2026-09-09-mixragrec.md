---
title: "[Paper Review] Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation (KDD 2026)"
excerpt: "이번 KDD 2026에 Accept된 MixRAGRec에 대한 논문 리뷰입니다."
categories:
  - recommend
tags:
  - [recommend]

permalink: /recommend/mixragrec/

toc: true
toc_sticky: true

date: 2026-09-09
last_modified_at: 2026-09-09
---

## Problem Statement

### Task : Knowledge-Augmented LLM-based Recommendation
Knowledge-Augmented LLM-based Recommendation은 사용자의 의도 (요구사항) 및 이전 상호작용 기록과 후보 아이템들이 주어졌을 때, 외부 정보 (Ex, 지식 그래프(KG))에서 관련된 지식을 검색하여 후보 아이템 중에서 사용자가 가장 선호할 만한 아이템을 추천하는 태스크입니다.

<img width="682" height="219" alt="image" src="https://github.com/user-attachments/assets/40e7c414-dca5-4e17-b247-a418ff18fee7" />


- 입력 : 사용자 쿼리 + 추천 후보 아이템 집합
  - 사용자 쿼리에는 사용자의 의도, Profile 등이 들어갈 수 있습니다. 이 정보들은 주로 사용자의 이전 상호작용 기록을 통해 만들어집니다.
- 출력 : 후보 아이템 중 선택한 추천 아이템

## Previous Limitations

기존의 Knowledge-Augmented Recommendation은 모든 사용자 질의에 동일한 Retrieval Granularity를 적용한다는 한계가 있습니다. 하지만 질의마다 추천에 필요한 외부 지식의 양과 구조가 다르기 때문에, 고정된 검색 방식은 다음과 같은 문제를 일으킬 수 있습니다.

<img width="678" height="437" alt="image" src="https://github.com/user-attachments/assets/ef265c47-6670-49e7-b03e-5effd117ee7f" />


- **Over-retrieval**
  - LLM만으로도 충분히 답할 수 있는 간단한 질의에도 KG를 검색하거나 큰 규모의 서브그래프를 가져옵니다.
  - 이로 인해 불필요한 노이즈가 추가되고 검색 및 추론 비용이 증가할 수 있습니다.
- **Under-retrieval**
  - 여러 엔티티 사이의 관계나 큰 서브그래프가 필요한 복잡한 질의에도 적은 양의 정보만 검색합니다.
  - 추천에 필요한 정보가 부족해져 최종 추천 성능이 감소할 수 있습니다.

또한 KG에서 검색한 결과는 그래프 구조이지만, LLM은 텍스트를 입력으로 받습니다. 그래프를 단순한 문자열이나 Triple 목록으로 변환하면 엔티티 사이의 구조적 관계가 손실되거나 추천에 불필요한 정보까지 포함될 수 있습니다.

마지막으로 각 질의에 어떤 Retrieval Granularity가 적절한지를 알려주는 정답 레이블이 존재하지 않습니다. 따라서 최종 추천 결과를 바탕으로 검색 방식의 적절성을 역으로 학습해야 하며, **Retrieval → Knowledge Conversion → Recommendation**의 세 단계를 함께 최적화하는 과정이 필요합니다.

이러한 문제를 해결하기 위해 본 논문에서는 Multi-Agent 기반의 Knowledge-Augmented LLM-based Recommendation 모델인 **MixRAGRec**을 제안합니다.

## Proposed Method : MixRAGRec

<img width="994" height="529" alt="image" src="https://github.com/user-attachments/assets/f8e7ac9a-4ee4-48ef-a05d-28829bcc1e10" />

MixRAGRec은 서로 다른 역할을 수행하는 세 에이전트를 **Mixture-of-Experts Multi-Agent Policy Optimization(MMAPO)**으로 공동 최적화하는 Multi-Agent LLM-based Recommendation 모델입니다.

- Input : 사용자의 질의(상호작용 이력 포함)와 후보 아이템 집합
- Output : 후보 아이템 중 사용자가 가장 선호할 것으로 예측한 아이템

전체 추천 과정은 다음과 같습니다.

1. Retrieval Agent가 사용자 질의에 적합한 KG Retrieval Expert를 선택합니다.
2. 선택된 Expert가 KG에서 관련 지식을 검색합니다.
3. Knowledge Preference Alignment Agent가 검색 결과를 자연어로 변환하고 추천에 필요한 정보만 남깁니다.
4. Recommendation Agent가 사용자 질의, 정렬된 지식, 후보 아이템을 바탕으로 최종 아이템을 추천합니다.

### Mixture-of-Experts Retrieval Agent
 
<img width="502" height="260" alt="image" src="https://github.com/user-attachments/assets/f85e32ea-c1c9-48d0-a9fb-4e2aeef20957" />

Mixture-of-Experts Retrieval Agent는 아이템 중심의 KG에서 구조화된 지식을 검색하는 단계입니다. 사용자 질의마다 필요한 KG 정보의 양과 구조가 다르기 때문에, 서로 다른 Granularity를 가진 네 가지 KG Retrieval 방식을 Expert로 구성하고 질의에 따라 하나를 선택합니다.

#### KG Construction and Indexing

먼저 사전에 구축한 KG를 다음과 같이 정의합니다.

$$
G = (\mathcal{V}, \mathcal{R}, \mathcal{T})
$$

여기서 $\mathcal{V}$는 Entity 집합, $\mathcal{R}$은 Relation 집합, $\mathcal{T}$는 Triple 집합을 의미합니다. 이후 KG를 검색할 수 있도록 두 종류의 Vector Database를 구축합니다.

- **Entity Vector Database** : 각 Entity의 Textual Description을 Encoder에 통과시켜 임베딩으로 저장합니다.
- **Triple Vector Database** : Head, Relation, Tail의 Textual Description을 결합하고 임베딩으로 변환하여 저장합니다.

사용자 질의 역시 동일한 Encoder를 통과시켜 임베딩으로 변환하며, 이 임베딩을 Entity 또는 Triple과의 유사도 계산에 사용합니다.

#### Mixture-of-Experts KG Retrieval

Retrieval Agent는 사용자 질의를 분석한 뒤 다음 네 가지 Expert 중 하나를 선택합니다.

- **Expert 1 — DirectGenerator**
  - KG Retrieval을 수행하지 않고 LLM이 직접 추천합니다.
  - 외부 지식이 필요하지 않은 단순한 질의에서 불필요한 검색 비용과 노이즈를 줄입니다.
- **Expert 2 — TripleRetriever**
  - 사용자 질의와 Triple Embedding 사이의 Cosine Similarity를 계산합니다.
  - 유사도가 높은 Triple을 직접 검색하여 질의와 관련된 개별 Fact를 가져옵니다.
- **Expert 3 — SubgraphRetriever**
  - 몇 개의 독립적인 Triple만으로 해결하기 어렵고 Multi-hop Relation이 필요한 질의에 사용합니다.
  - 먼저 질의와 유사도가 높은 Entity를 Seed로 선택하고, 각 Seed를 중심으로 $K$-hop Neighborhood를 확장하여 Subgraph를 구성합니다.
- **Expert 4 — ConnectedGraphRetriever**
  - 복잡한 질의에 대해 전역 그래프 관점에서 일관성 있는 지식이 필요할 때 사용합니다.
  - 질의와 관련성이 높은 Top-$M$ Entity를 Seed로 선택한 뒤 Personalized PageRank(PPR)를 실행하고, PPR Score가 높은 Top-$M$ Entity를 다시 선택합니다.
  - 각 Edge의 Relation과 질의 사이의 유사도를 계산하며, 관련성이 높은 Edge일수록 낮은 Cost를 부여합니다.
  - 이후 Kruskal's Minimum Spanning Tree(MST)를 적용하여 중요 Entity를 연결하면서도 질의와 무관한 Edge는 최대한 제외한 Compact Connected Graph를 만듭니다. 

>인코더는 sentence-transformers/all-MiniLM-L6-v2 모델을 사용합니다.

### Knowledge Preference Alignment Agent

Knowledge Preference Alignment Agent는 검색된 KG를 먼저 자연어로 변환한 뒤, 사용자 질의를 조건으로 추천에 유용한 지식만 강조하도록 정제하는 단계입니다. KG를 검색하지 않는 DirectGenerator (Expert 1)가 선택된 경우에는 이 단계를 수행하지 않습니다.

먼저 검색 결과를 안정적인 Textual Draft로 변환합니다.

- **TripleRetriever의 검색 결과**
  - 각 Triple을 `{head} has {relation} {tail}`과 같은 Relation-aware Template으로 자연어화합니다.
  - 여러 Triple을 하나의 간결한 Knowledge Statement로 결합합니다.
- **SubgraphRetriever와 ConnectedGraphRetriever의 검색 결과**
  - 그래프를 단순한 Triple 목록으로 나열하지 않고, Entity 사이의 중요한 Relation이나 짧은 Path를 중심으로 Linearization합니다.
  - 예를 들어 `(Inception, director, Nolan)`, `(Nolan, director_of, Oppenheimer)`라는 두 Triple은 `Inception is directed by Christopher Nolan. Christopher Nolan also directed Oppenheimer.`와 같이 변환할 수 있습니다.

이후 Agent는 사용자 질의와 Textual Draft를 함께 입력받아 현재 추천에 필요한 정보가 무엇인지 판단하고, 노이즈를 제거한 **Aligned Knowledge Snippet**을 생성합니다. (정제한 자연어 형태의 지식 문장)

### Contrastive Learning-Reinforced Recommendation Agent

Recommendation Agent는 정답 아이템과 모델이 혼동하는 Hard Negative Item을 Pair로 구성하고, 동일한 사용자 Context에서 정답 아이템의 추천 확률이 Hard Negative보다 높아지도록 Preference Optimization을 수행합니다.

- Input : 사용자 질의, Alignment Agent가 정렬한 지식, 후보 아이템 집합
- Output : 후보 아이템별 추천 확률 및 최종 추천 아이템

학습 과정은 다음과 같습니다.

1. 후보 아이템 중 Ground Truth를 Target Item으로, 나머지를 Negative Item으로 구분합니다.
2. 현재 Recommendation Agent가 높은 확률을 부여한 Top-$N$ Negative Item을 Hard Negative로 선택합니다.
3. 각 Hard Negative에 대해 `(Context, Target, Hard Negative)` 형태의 Preference Pair를 구성합니다.
4. Target Item의 추천 확률이 Hard Negative보다 높아지도록 Preference Optimization을 수행합니다.
5. Hard Negative가 여러 개라면 각 Pair의 Preference Loss를 계산한 뒤 평균하여 Recommendation Agent를 학습합니다.

### Mixture-of-Experts Multi-Agent Policy Optimization (MMAPO)

Retrieval, Knowledge Alignment, Recommendation은 서로 영향을 주기 때문에 각 Agent의 기여도를 개별적으로 판단하기 어렵습니다. MMAPO는 최종 추천 성능과 KG Retrieval의 정보적 유용성 및 비용을 함께 반영한 Shared Reward를 사용하여 세 Agent를 Coordinated하게 학습합니다.

전체 과정은 다음과 같습니다.

1. Retrieval Agent가 사용자 질의를 바탕으로 네 가지 Expert 중 하나를 선택합니다.
2. 선택된 Expert로 KG를 검색한 뒤 Alignment Agent가 Aligned Knowledge를 생성합니다.
3. Recommendation Agent가 사용자 질의, Aligned Knowledge, 후보 아이템을 이용해 최종 아이템을 추천합니다.
4. 최종 결과를 바탕으로 Recommendation Reward와 Marginal Information Gain(MIG) Reward를 계산합니다.
5. 최종 Reward를 Retrieval 및 Alignment 단계의 이전 Action에 할당하여 각 Agent를 학습합니다.

Retrieval Agent와 Alignment Agent는 Advantage 기반의 PPO-style Policy Optimization으로 학습하고, Recommendation Agent는 Target과 Hard Negative 사이의 Contrastive Preference Optimization으로 학습합니다.

#### Advantage Estimation

Expert 선택의 효과는 최종 추천 이후에 확인할 수 있으므로, MMAPO는 GAE(Generalized Advantage Estimation)를 사용해 최종 Reward를 이전 Action에 Credit Assignment합니다.

- 각 중간 Step에는 즉시 Reward를 부여하지 않고, 마지막 Step에서만 $R_{total}$을 Terminal Reward로 사용합니다.
- Retrieval Agent와 Alignment Agent는 각각 Value Function을 통해 현재 State에서 기대되는 Future Reward를 추정합니다.
- 각 Step에서 Temporal-Difference Residual을 계산하고, 이후 Step에서 발생한 TD Error를 Discount하여 합산함으로써 GAE를 계산합니다.
- 계산된 Advantage를 이용하여 좋은 결과에 기여한 Expert 선택과 Alignment Action의 확률은 높이고, 좋지 않은 결과에 기여한 Action의 확률은 낮춥니다.

이를 통해 Retrieval Agent에는 선택한 Expert가 최종 추천에 얼마나 도움이 되었는지를, Alignment Agent에는 생성한 Knowledge 또는 Token이 최종 추천에 얼마나 기여했는지를 학습 신호로 전달합니다.

구체적으로 과정을 분할하면 다음 2가지로 구분할 수 있습니다. 

#### Reward Design

Shared Reward는 다음 두 요소로 구성됩니다.

- **Recommendation Reward($R_{rec}$)**
  - Recommendation Agent가 Ground Truth Item을 올바르게 추천했는지를 평가합니다.
  - 정답을 추천한 경우 Recommendation Confidence에 기반한 Positive Reward를 부여하고, 잘못 추천한 경우에는 $-0.1$의 Negative Reward를 부여합니다.
- **Marginal Information Gain Reward($R_{MIG}$)**
  - 선택한 KG Retrieval Expert가 추천에 실제로 유용한 정보를 제공했는지를 평가합니다.
  - KG Retrieval을 사용한 경우와 DirectGenerator를 사용한 경우의 Recommendation Distribution 차이를 KL Divergence로 계산합니다.
  - Retrieval Cost를 Penalty로 함께 반영하여, 추가 비용보다 충분히 큰 Information Gain을 제공할 때만 높은 Reward를 받을 수 있도록 합니다.

이 Reward Design은 정확한 추천을 유지하면서도 불필요하게 복잡하고 비용이 큰 KG Retrieval을 피하도록 유도합니다.

## Experiments

### Experimental Settings

- KG : DBpedia를 활용하여 구축
- LLM Backbone : LLaMA3-8B, Mistral-7B
- Fine-tuning : LoRA Fine-tuning
- Hardware : NVIDIA H20-96GB GPU 1장
- User Context : 마지막 아이템을 Target Item으로 설정하고, 그 직전 10개의 Interaction을 사용
- Candidate Pool : 20개
- Accuracy : 20개의 후보 중 Ground Truth Item을 Top-1으로 예측한 비율

### Baselines

| Category | Baseline |
| --- | --- |
| Zero-shot LLM | Gemini-2.0-Flash, GPT-4o, DeepSeek-R1, LLaMA3-8B, Mistral-7B |
| LLM-based Recommendation | TallRec, Rec-r1 |
| KG-RAG Recommendation | KG-Text, KAPING, G-Retriever, K-RagRec |

<img width="967" height="584" alt="image" src="https://github.com/user-attachments/assets/d3ea86cf-de30-40f0-9ed2-7fed1eca160d" />


### Ablation Study

<img width="529" height="729" alt="image" src="https://github.com/user-attachments/assets/5ec47e28-c840-4497-96ba-3137b9cc89d3" />

| Variant | Removed or Modified Component |
| --- | --- |
| `w/o MoESel` | 학습된 Expert Selector를 Random Expert Selection으로 변경 |
| `w/o Align` | Alignment Agent를 제거하고 Template Text만 사용 |
| `w/o HardNeg` | Hard Negative를 Random Negative로 변경 |
| `w/o PrefOpt` | Contrastive Preference Optimization 제거 |
| `w/o MMAPO` | Shared Objective와 MIG를 제거하고 각 Agent를 개별적으로 학습 |

Ablation Study에서는 평균적으로 Alignment Agent를 제거했을 때의 성능 저하가 크게 나타났습니다. 이는 그래프에서 관련 정보를 검색하는 것뿐만 아니라, 검색 결과에 포함된 노이즈를 줄이고 추천에 필요한 형태로 정확하게 정렬하는 과정이 중요하다는 점을 보여줍니다.
