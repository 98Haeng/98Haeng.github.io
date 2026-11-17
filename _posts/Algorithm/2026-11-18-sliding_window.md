---
title: "[DS&AL] Sliding Window"
excerpt: "자료구조의 유형 중 Two Pointer에서 자주 활용되는 Sliding Window에 대한 내용입니다."

categories:
  - Data Structure & Algorithm
tags:
  - [Data Structure & Algorithm]

permalink: /dsal/slidingwindow/

toc: true
toc_sticky: true

date: 2026-11-18
last_modified_at: 2026-11-18
---

# Two Pointer
Sliding Window(슬라이딩 윈도우)는 **배열이나 문자열의 연속된 구간(Window)을 유지하면서, 구간을 이동해 원하는 값을 찾는 기법**입니다.

투 포인터(Two Pointer)의 한 유형으로, `left`, `right`로 구간을 표현합니다. 중요한 점은 포인터만 움직이는 것이 아니라, **구간이 바뀔 때 새로 들어오는 원소와 빠지는 원소만 반영**한다는 것입니다.

```text
배열: [1, 2, 3, 4, 5, 6]

[1, 2, 3] 4  5  6    ← 첫 번째 Window
 1 [2, 3, 4] 5  6    ← 오른쪽으로 한 칸 이동
 1  2 [3, 4, 5] 6
```

슬라이딩 윈도우의 핵심은 구간이 이동할 때마다 전체 구간을 다시 계산하지 않고, 새롭게 들어오는 값과 빠지는 값만 반영하는 것입니다.

예를 들어 길이가 3인 구간의 합을 계산할 때,
- 기존 구간: [1, 2, 3] → 합 6
- 다음 구간: [2, 3, 4] → 기존 합에서 1을 빼고 4를 더해 합 9

이처럼, 이전 구간의 계산 결과를 재사용하게 됩니다.

# 문제 유형

슬라이딩 윈도우는 크게 **고정 크기**, **가변 크기**, **2차원 구간**으로 나누어 접근할 수 있습니다.

### 1. 고정 크기 Sliding Window

Window의 길이가 `K`로 정해져 있고, 그 크기를 유지하며 한 칸씩 이동하는 유형입니다.

```text
배열: [1, 2, 3, 4, 5, 6]
K = 3

[1, 2, 3] 4  5  6
 1 [2, 3, 4] 5  6
 1  2 [3, 4, 5] 6
 1  2  3 [4, 5, 6]
```

여기서 나올 수 있는 대표 문제는 다음과 같습니다.

- 길이가 K인 연속 부분 배열의 최대/최소 합
- 길이가 K인 구간의 평균
- 고정 길이 구간의 최대/최소 원소
- 특정 크기의 구간에서 조건을 만족하는 원소 개수

예를 들어, 길이가 K인 연속 부분 배열의 최대 합을 구하는 문제라면 다음과 같이 구현할 수 있습니다.

```python
arr = [1, 2, 3, 4, 5, 6]
k = 3

window_sum = sum(arr[:k])
answer = window_sum

for right in range(k, len(arr)):
    window_sum += arr[right]      # 오른쪽 값 추가
    window_sum -= arr[right-k]    # 왼쪽 값 제거

    answer = max(answer, window_sum)

print(answer)
```
### 접근 방법: 구간의 최솟값·최댓값 유지

구간 합과 달리, 최솟값·최댓값은 원소 하나를 제거했다고 단순한 사칙연산으로 갱신할 수 없습니다. 이때는 **Monotonic Deque(단조 큐)** 를 활용합니다.

최솟값을 구하는 경우 Deque에는 **값이 증가하는 순서가 되도록 인덱스**를 저장합니다.

```python
from collections import deque

arr = [4, 2, 7, 3, 1]
k = 3
queue = deque()  # 값이 아니라 인덱스를 저장

for right in range(len(arr)):
    # 1. 현재 구간을 벗어난 인덱스 제거
    while queue and queue[0] <= right - k:
        queue.popleft()

    # 2. 새 원소보다 크거나 같은 값은 후보에서 제거
    while queue and arr[queue[-1]] >= arr[right]:
        queue.pop()

    # 3. 새 원소의 인덱스 추가
    queue.append(right)

    # 4. 길이 K인 구간이 완성된 후 최솟값 확인
    if right >= k - 1:
        current_min = arr[queue[0]]
        # current_min을 이용해 문제에서 요구하는 값 갱신
```

최댓값을 구하려면 두 번째 `while`의 비교를 `<=`로 바꿉니다. 각 원소가 Deque에 최대 한 번 들어가고 한 번 빠지므로 **전체 O(N)** 에 처리할 수 있습니다.

### 2. 가변 크기 Sliding Window

Window의 길이가 고정되지 않고, **현재 구간이 조건을 만족하는지**에 따라 `left`, `right`를 움직이는 유형입니다.

```text
배열: [A, B, A, C, B]
조건: A, B, C를 모두 포함

[A, B, A] C  B    ← C가 없으므로 확장
[A, B, A, C] B    ← 조건을 만족
 A [B, A, C] B    ← 조건을 유지하면서 축소
```
대표적으로 **모든 종류를 포함하는 최소 구간, 합이 일정 값 이상인 최소 길이 구간(양수 배열), 중복 없는 가장 긴 부분 문자열**을 찾는 문제가 있습니다.

### 접근 방법 A: 조건을 만족하면 왼쪽 축소

**최소 길이 구간**을 구할 때 자주 사용하는 방식입니다.

1. `right`를 이동하며 원소를 추가합니다.
2. 구간이 조건을 만족하면 현재 구간을 정답 후보로 확인합니다.
3. 조건을 만족하는 동안 `left`를 이동해 구간을 줄입니다.
4. 조건이 깨지면 다시 `right`를 확장합니다.

예를 들어 모든 종류의 보석을 포함하는 구간이라면, `dict`로 **구간 내 보석별 개수**를 관리합니다.

```python
gems = ["A", "B", "A", "C", "B"]
required = len(set(gems))

left = 0
bucket = {}  # 현재 Window에 있는 보석 종류별 개수

for right in range(len(gems)):
    # 1. 오른쪽 보석 추가
    gem = gems[right]
    bucket[gem] = bucket.get(gem, 0) + 1

    # 2. 조건을 만족하는 동안 왼쪽 축소
    while len(bucket) == required:
        # 현재 [left, right]를 정답 후보로 확인

        left_gem = gems[left]
        bucket[left_gem] -= 1

        if bucket[left_gem] == 0:
            del bucket[left_gem]

        left += 1
```

`set(gems[left:right+1])`을 매번 계산하면 구간 전체를 다시 읽게 됩니다. 반면 `dict`로 개수를 유지하면 **원소 추가·제거가 평균 O(1)** 이고, 두 포인터가 각각 최대 N번 이동해 **전체 O(N)** 이 됩니다.

### 접근 방법 B: 조건을 위반하면 왼쪽 축소

반대로 **최대 길이 구간**을 찾는 문제에서는 조건을 만족하는 동안 확장하다가, 조건이 깨졌을 때만 왼쪽을 줄이는 방식이 자주 사용됩니다.

예를 들어 **중복 문자가 없는 가장 긴 부분 문자열**이라면 다음과 같은 구조로 접근합니다.

```python
s = "abac"
left = 0
bucket = {}

for right in range(len(s)):
    # 1. 오른쪽 문자 추가
    char = s[right]
    bucket[char] = bucket.get(char, 0) + 1

    # 2. 중복이 생기면 중복이 없어질 때까지 왼쪽 축소
    while bucket[char] > 1:
        left_char = s[left]
        bucket[left_char] -= 1
        if bucket[left_char] == 0:
            del bucket[left_char]
        left += 1

    # 3. 현재 [left, right]는 중복 없는 구간
    # 길이(right - left + 1)를 정답 후보로 확인
```

즉, 가변 크기 유형에서는 다음을 먼저 판단하면 됩니다.

- **최소 길이**를 찾는다면: 조건을 만족할 때 가능한 만큼 축소할 수 있는가?
- **최대 길이**를 찾는다면: 조건을 위반할 때까지 확장하고, 위반하면 축소할 수 있는가?

단, 이런 이동 방식은 구간의 조건이 포인터 이동에 따라 적절하게 유지되는 문제에서 적용할 수 있습니다. 예를 들어 부분합을 기준으로 하는 가변 윈도우는 일반적으로 **양수(또는 조건에 맞는 비음수) 원소**라는 전제가 필요합니다.

## 3. 2차원 Sliding Window

2차원 행렬에서 `H × W` 크기의 직사각형 구간을 이동시키는 유형입니다.

(대표적으로 프로그래머스의 선인장 숨기기 문제가 포함됩니다. [선인장 숨기기](https://school.programmers.co.kr/learn/courses/30/lessons/468379))
```text
3 × 4 행렬, Window = 2 × 2

Step 1               Step 2
[1  2] 3  4           1 [2  3] 4
[5  6] 7  8           5 [6  7] 8
 9 10 11 12           9 10 11 12
```

대표적으로 **H × W 구간의 합, 최솟값·최댓값, 각 구간의 최솟값 중 최댓값**을 구하는 문제가 있습니다.

### 접근 방법: 가로 → 세로로 나누어 처리

구간의 **최솟값·최댓값**을 구하는 경우, 2차원 구간을 한꺼번에 탐색하는 대신 1차원 슬라이딩 윈도우를 두 번 적용합니다.

1. 각 행에서 가로 길이 `W`인 구간의 최솟값을 구합니다.
2. 가로 결과 배열의 각 열에서 세로 길이 `H`인 구간의 최솟값을 구합니다.
3. 이렇게 얻은 값이 원본 행렬의 각 `H × W` 구간의 최솟값입니다.

```text
원본 행렬 (3 × 4)
4  2  7  3
6  1  5  8
9  4  2  6

① 가로 길이 W=2인 구간의 최솟값
2  2  3
1  1  5
4  2  2

② 세로 길이 H=2인 구간의 최솟값
1  1  3
1  1  2
```

위의 **가로/세로 단계 모두 동일한 단조 Deque 패턴**으로 구현할 수 있습니다. 아래 코드는 특정 문제의 최종 정답을 구하는 대신, **모든 H × W 구간의 최솟값 배열을 만드는 기본 구조**입니다.

```python
from collections import deque

# layers: M × N 행렬
m, n = len(layers), len(layers[0])

# 1. 가로 방향: 각 행에 길이 W인 Sliding Window 적용
row_min = [[0] * (n - w + 1) for _ in range(m)]

for i in range(m):
    queue = deque()

    for j in range(n):
        while queue and queue[0] <= j - w:
            queue.popleft()

        while queue and layers[i][queue[-1]] >= layers[i][j]:
            queue.pop()

        queue.append(j)

        if j >= w - 1:
            row_min[i][j - w + 1] = layers[i][queue[0]]

# 2. 세로 방향: 가로 결과의 각 열에 길이 H인 Window 적용
window_min = [[0] * (n - w + 1) for _ in range(m - h + 1)]

for j in range(n - w + 1):
    queue = deque()

    for i in range(m):
        while queue and queue[0] <= i - h:
            queue.popleft()

        while queue and row_min[queue[-1]][j] >= row_min[i][j]:
            queue.pop()

        queue.append(i)

        if i >= h - 1:
            window_min[i - h + 1][j] = row_min[queue[0]][j]

# window_min: 각 H × W 구간의 최솟값
# 이후 문제의 조건에 맞게 결과 활용
```

이 방식은 각 행·열을 한 번씩 처리하므로 전체 시간 복잡도가 **O(MN)** 입니다. 다만 **구간 합**을 구하는 문제라면 2차원 누적 합(Prefix Sum)을 사용하는 방법도 고려할 수 있습니다.


# 유형별 정리

| 유형 | Window 크기 | 이동 방식 | 주로 유지할 정보 | 일반적인 시간 복잡도 |
|---|---|---|---|---|
| 고정 크기 — 합/개수 | 고정 `K` | 들어오는 값 추가, 빠지는 값 제거 | 합, 개수 | O(N) |
| 고정 크기 — 최솟값/최댓값 | 고정 `K` | 범위 밖 원소·불필요한 후보 제거 | Monotonic Deque | O(N) |
| 가변 크기 — 최소 구간 | 변동 | 조건 충족 시 `left` 축소 | 합, `dict` 등 | O(N) (조건 충족 시) |
| 가변 크기 — 최대 구간 | 변동 | 조건 위반 시 `left` 축소 | `dict`, 빈도 등 | O(N) (조건 충족 시) |
| 2차원 — 구간 최솟값/최댓값 | 고정 `H × W` | 가로 적용 후 세로 적용 | Monotonic Deque | O(MN) |

**정리하면**, 슬라이딩 윈도우 문제를 볼 때는 **① Window 크기가 고정인지, ② 언제 확장·축소할지, ③ 구간의 어떤 상태를 유지할지**를 먼저 결정하면 됩니다. 구간 전체를 다시 확인하는 대신, 이동으로 달라진 부분만 갱신하는 것이 공통된 접근 방식입니다.
