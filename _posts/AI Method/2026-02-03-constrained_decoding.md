---
title: "[AI Method] Constrained Decoding"
excerpt: "LLM에서 주로 사용되는 Constrained Decoding에 대한 설명입니다."

categories:
  - AI Method
tags:
  - [AI Method]

permalink: /aimethod/constrained-decoding/

toc: true
toc_sticky: true

date: 2026-02-03
last_modified_at: 2026-02-03
---

# Constrained Decoding

Constrained Decoding은 자연어 생성 작업에서 생성된 텍스트가 제약 조건을 만족하도록 보장하는 디코딩 방법입니다. 언어 모델이 다음 토큰을 고를 때, **허용되는 출력 집합**을 강제로 제한합니다.

모델이 아무 말이나 생성하는 것 대신, 
- 허용하는 문자열이나 토큰만 생성되도록 하거나,
- 정해진 문법 or 형식을 만족하는 방향으로만 생성되게 하거나
- 후보 리스트 중에서만 선택하도록

제한을 두는 방식입니다.

Constrained Decoding에는 대표적인 패턴이 3가지가 있습니다.
- Token-level 제약 : 다음 토큰은 해당 집합 안에서만 선택 가능하도록 설정합니다.
- Prefix(문자열) 기반 제약 (Trie 제약) : 가장 많이 쓰는 방법으로써, 최종 출력을 허용된 문자열 목록 중 하나로 제한합니다
  - 각 허용 문자열을 토크나이징해서 Trie를 만들고, 현재까지 prefix에 따라 다음에 올 수 있는 토큰만 허용합니다.
- Grammer/Regex/Schema/FSM 등 제약 : Json Schema / 정규식 / CFG 등에 맞게 토큰을 마스킹합니다.

## Trie + Prefix 기반 제약 예시
Huggingface의 transformers의 generate()의 prefix_allowed_token_fn을 사용해 모델 출력이 candidate 중 하나가 되도록 강제합니다.
하기 코드에서는 "artificial intelligence", "machine learning", "deep learning framework" 이 3개 중 하나로 답을 출력하도록 강제하고 있습니다.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 모델과 토크나이저 로드
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. 강제하고 싶은 후보 단어 리스트 (여기에 있는 것만 생성 가능)
candidates = [
    "artificial intelligence",
    "machine learning",
    "deep learning framework"
]

# 3. Trie(접두사 트리) 구축 함수 : 후보 단어들을 토큰화하여 트리 구조로 만듭니다.
def build_trie(candidates, tokenizer):
    trie = {}
    for word in candidates:
        # 각 단어를 토큰 ID 리스트로 변환
        token_ids = tokenizer.encode(word)
        node = trie
        for tid in token_ids:
            if tid not in node:
                node[tid] = {}
            node = node[tid]
        # 문장의 끝임을 표시 (End of Sentence)
        node[-1] = True 
    return trie

# Trie 생성
trie_root = build_trie(candidates, tokenizer)

# 4. 핵심: prefix_allowed_tokens_fn 정의
# 이 함수는 매 생성 스텝마다 호출됩니다.
def restrict_decode_vocab(batch_idx, input_ids):
    """
    batch_idx: 현재 배치의 인덱스 (int)
    input_ids: 현재까지 생성된 전체 토큰 시퀀스 (tensor)
    return: 다음에 생성 허용할 토큰 ID 리스트 (list[int])
    """
    # 텐서를 리스트로 변환
    prefix = input_ids.tolist()
    
    # 프롬프트: "I am studying " (4토큰이라고 가정)
    curr_node = trie_root
    
    # 이미 생성된 토큰들을 따라 Trie를 내려갑니다.
    # (실제로는 프롬프트 이후의 토큰만 순회해야 합니다)
    tokens_generated_so_far = prefix[len(prompt_ids):]
    
    for token in tokens_generated_so_far:
        if token in curr_node:
            curr_node = curr_node[token]
        else:
            # Trie 경로를 벗어난 경우 (이미 생성 실패 or 끝남) -> 더 이상 생성하지 않도록 EOS 토큰 반환
            return [tokenizer.eos_token_id]    
    # 현재 노드에서 갈 수 있는 다음 토큰 ID들을 반환
    allowed_next_tokens = [key for key in curr_node.keys() if key != -1]
    # 만약 단어가 끝났다면(-1 존재), EOS 토큰도 허용 목록에 추가
    if -1 in curr_node:
        allowed_next_tokens.append(tokenizer.eos_token_id)
    return allowed_next_tokens

# 5. 실행
prompt = "I am studying "
prompt_ids = tokenizer.encode(prompt)

# 입력을 텐서로 변환
input_tensor = torch.tensor([prompt_ids])

output = model.generate(
    input_tensor,
    max_new_tokens=10,
    # 여기서 함수를 주입합니다.
    prefix_allowed_tokens_fn=restrict_decode_vocab,
    pad_token_id=tokenizer.eos_token_id
)

print("Output:", tokenizer.decode(output[0]))
# 결과는 모델 확률과 무관하게 candidates 중 하나로 완성됩니다.
# 예: "I am studying artificial intelligence"
```

코드에 대한 설명을 하나씩 드리면, 
```python
def build_trie(candidates, tokenizer):
    trie = {}
    for word in candidates:
        # 각 단어를 토큰 ID 리스트로 변환
        token_ids = tokenizer.encode(word)
        node = trie
        for tid in token_ids:
            if tid not in node:
                node[tid] = {}
            node = node[tid]
        # 문장의 끝임을 표시 (End of Sentence)
        node[-1] = True 
    return trie
```
여기에서 사용되는 Trie는 **지금까지 생성된 토큰 prefix**을 보고, 다음에 올 수 있는 토큰 후보를 빠르게 찾는 구조입니다. 여기서 Trie는 dictionary 형태로 표현했습니다. 
- Trie root를 {token_id: {token_id: {...}}} 형태로 표현하고,
- 후보 문자열을 토큰화한 시퀀스를 하나의 경로로 삽입합니다.
- 후보 문자열의 끝에는 ```node[-1]=True```를 삽입하여 여기서 문장이 끝날 수 있음을 보여줍니다.

후보를 토큰화하는 과정에서는 위 코드에서는 GPT-2를 예시로 사용했는데, 여기서는 단어 단위가 아닌, 서브워드 단위 토큰으로 분해됩니다. 예를 들어 설명하면, "machine learning"이란 단어가 [12345, 65] 이런 형식으로 여러 토큰으로 분할됩니다.

다음으로, prefix_allowed_tokens_fn 부분에 입력되는 내용을 살펴보면, generate()에서 사용되는데, 지정된 리스트 (Trie에 없는 경우)가 아닌 경우에는 생성을 막습니다. 이후 과정은, prefix가 **프롬프트 토큰 + 지금까지 생성된 토큰**이 같이 들어있는데, 프롬프트 뒤에 생성된 부분만 Trie로 검사합니다. 그래서, 프롬프트 길이만큼 잘라서 tokens_generated_so_far로 만들어줍니다. 

그 다음, 

```python
curr_node = trie_root
for token in tokens_generated_so_far:
    if token in curr_node:
        curr_node = curr_node[token]
    else:
        return [tokenizer.eos_token_id]
```
이 코드에서 지금까지 생성된 토큰들이 Trie의 경로를 잘 따라가고, ```curr_node``` 변수는 현재 prefix 위치를 가리킵니다. 여기서 만약 경로를 벗어나게 된다면 (즉, 후보 문자열의 prefix를 생성할 수 없다면), 더 이상 올바른 후보로 완성될 수 없기에 EOS만 허용하여 종료시킵니다.

```python
allowed_next_tokens = [key for key in curr_node.keys() if key != -1]
if -1 in curr_node:
        allowed_next_tokens.append(tokenizer.eos_token_id)
    return allowed_next_tokens
```
다음 해당 토큰에서는, 현재 노드에서 가능한 **Next token**을 반환합니다. 그리고 만약 현재 노드가 문장이 끝낼 수 있는 위치라면, EOS를 가능하게 하여 깔끔하게 종료시키는 구조입니다.

복잡할 수 있지만, 간단하게 말하면, 매 스텝마다 허용 토큰을 Trie가 결정합니다. 즉, 다른 단어를 사용하거나, 선택지가 없는 경우 허용 목록에 없어서 선택 불가, 즉 종료합니다.

## 허용 집합 외 확률 마스킹 방법 예시
확률을 마스킹하는 부분은 다음과 같은 예시 코드로 작동합니다.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor

class AllowTokenIdsProcessor(LogitsProcessor):
    def __init__(self, allow_ids: List[int]):
        self.allow = torch.tensor(sorted(set(allow_ids)), dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores shape: (batch, vocab)
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allow] = 0.0
        return scores + mask

# 예시: causal LM
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# "0"~"9"가 단일 토큰으로 존재한다는 가정은 모델마다 다를 수 있어,
# 실제로는 tokenizer로 인코딩해서 얻은 토큰 id를 모읍니다.
allow_ids = []
for d in list("0123456789"):
    ids = tokenizer.encode(d, add_special_tokens=False)
    if len(ids) == 1:
        allow_ids.append(ids[0])

# 필요하면 공백/줄바꿈/eos 등도 허용
if tokenizer.eos_token_id is not None:
    allow_ids.append(tokenizer.eos_token_id)

processor = AllowTokenIdsProcessor(allow_ids)

prompt = "Output a 5-digit number: "
inputs = tokenizer(prompt, return_tensors="pt")

out = model.generate(
    **inputs,
    max_new_tokens=8,
    do_sample=False,
    logits_processor=[processor],
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

해당 코드에서는 ```logits_processor```을 이용해서 모델이 내놓은 점수를 수정하는 방법입니다. 그 이후, 수정된 점수로 다음 토큰을 선택하는 방법입니다.

