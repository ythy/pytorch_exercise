## 数据训练变化演示

### 测试数据和关键代码
```python
x_train = tensor([[ 1.8741,  1.4317,  0.6764, -1.4240, -0.6000, -1.6977, -1.6977],
        [ 1.8741,  1.4317,  1.3884,  1.0072,  0.8999, -1.2127, -1.2127],
        [ 0.3123,  0.5895, -1.1036, -0.7294, -1.1624, -0.7276, -0.7276],
        [-0.4685,  0.5895, -0.7476,  1.7018, -0.0375, -0.2425, -0.2425],
        [-0.4685, -0.2526, -0.7476, -0.0347, -0.9124,  0.2425,  0.2425],
        [-0.4685, -1.0948, -0.7476, -1.0767, -0.3500,  0.2425,  0.7276],
        [-0.4685, -1.0948,  1.0324, -1.0767, -1.4124,  0.7276,  0.2425],
        [-0.4685, -1.0948,  0.6764,  0.3126,  1.3374,  1.2127,  1.2127],
        [-0.4685, -1.0948,  1.0324,  0.3126,  0.7749, -0.2425,  1.6977],
        [-1.2494,  0.5895, -1.4596,  1.0072,  1.4624,  1.6977, -0.2425]])
features = [
    "Overall", "Potential", "Age", "height_cm", "weight_kg",
    "Nationality_enc", "Position_enc"
]
model = nn.Sequential(
    nn.Linear(len(features), 128), # W初始是很小默认生成的随机数; h1 = [2.1, -0.3, 0.8, -1.5, 0.4, ...]  (128维)
    nn.ReLU(), #Rectified Linear Unit；  ReLU(x)=max(0,x) 负信号：丢弃， 正信号：保留
    nn.Linear(128, 64), # h2_relu = [0, 1.5, 0, 0.9, ...]   (64维)
    nn.ReLU(),
    nn.Linear(64, 1)  
)
y_pred = model(x_train)
```

### 分析已知条件
```python
#10条数据7个特征
x_train.shape = (10, 7) 
# 线性 输入7个特征，输出128个特征, PyTorch 内部定义：
# self.weight.shape == (128, 7) 随机的128组7项特征权重组合
# self.bias.shape == (128,) b在计算后确实会变成 (10, 128)，但它的“参数形状”永远是 (128,)，靠 broadcasting 自动对齐。
nn.Linear(7, 128) 
# 实际数学角度运算 (10, 128)  输出正确
y = x @ W.T + b
x.shape == (10, 7)
W.shape == (128, 7)
W.T.shape == (7, 128) #随机的128列7项特征权重组合
(10, 7) @ (7, 128) + (10, 128) → (10, 128) 

nn.Linear(128, 64)
y = x @ W.T + b
x.shape == (10, 128)
W.shape == (64, 128)
W.T.shape == (128, 64) #随机的64列128项特征权重组合
(10, 128) @ (128, 64) + (10, 64) → (10, 64) 

nn.Linear(64, 1)
y = x @ W.T + b
x.shape == (10, 64)
W.shape == (1, 64)
W.T.shape == (64, 1) #随机的1列64项特征权重组合
(10, 64) @ (64, 1) + (10, 1) → (10, 1) 
```
 

## 一些概念
### nn.Sequential
线性堆叠模型容器​，用于实现MLP, MLP 可以用 Sequential 写，也可以用 Module 写
```
input → layer1 → layer2 → ... → output
```
### MLP
MLP = Multilayer Perceptron Regression(回归), 模型类型
> 由多个全连接层（Linear）+ 非线性激活组成的神经网络
```
x → Linear → ReLU → Linear → ReLU → ... → 输出
```

## Transformer 的 Self-Attention
> Query 是“我想查什么”，Key 是“我有什么”，Value 是“我提供什么信息”

1️⃣ Query（查询）：「我在关注什么」
每一个 token 会生成一个 Query 向量
表示：“当前这个 token，想要去和其他 token 建立联系时的‘提问方式’”
2️⃣ Key（键）：「我可以被怎样匹配」
每一个 token 也会生成一个 Key 向量
表示：“我这里有什么特征，可以用来被别的 token 匹配”
3️⃣ Value（值）：「我真正携带的信息」
每一个 token 生成一个 Value 向量
表示：“如果我被关注到，我贡献什么内容”

### 案例
####  输入 X（3 个 token，维度 = 4）
```python
X =
[
  [1, 0, 0, 0],   // token1
  [0, 1, 0, 0],   // token2
  [0, 0, 1, 0]    // token3
]
```
#### 参数矩阵（可学习）
```python
W_Q =
[
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1]
]
W_K =
[
  [1, 0],
  [1, 0],
  [0, 1],
  [0, 1]
]
W_V =
[
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 0]
]
```
#### 计算 Q / K / V
`Query Q = X · W_Q;  Key K = X · W_K;  Value V = X · W_V`
```python
Q =
[
  [1, 0],   // q1
  [0, 1],   // q2
  [0, 1]    // q3
]
K =
[
  [1, 0],   // k1
  [1, 0],   // k2
  [0, 1]    // k3
]
Kᵀ =
[
  [1, 1, 0],   // Kᵀ1
  [0, 0, 1],   // Kᵀ2
]
V =
[
  [1, 0, 0],   // v1
  [0, 1, 0],   // v2
  [1, 0, 0]    // v3
]
```
#### Attention Score（Q · Kᵀ）
Kᵀ 是 Key 矩阵的转置（transpose）
```python
Scores =
[
  [1, 1, 0],
  [0, 0, 1],
  [0, 0, 1]
]·
```
#### Softmax（按行，省略缩放）
Softmax 保证一行数据 非负，和为1，保持原先的大小关系
```python
Weights =
[
  [0.42, 0.42, 0.16], // token1 的注意力分布
  [0.23, 0.23, 0.54], // token2
  [0.23, 0.23, 0.54]  // token3
]
```
#### 输出 Z（加权 Value）
Z = Weights · V
```python
Z =
[
  [0.58, 0.42, 0],
  [0.77, 0.23, 0],
  [0.77, 0.23, 0]
]
```
#### Loss = f(Z) 反向传播
相当于loss.backward()
“Loss 对 Z 的偏导”
∂L：Loss 的无穷小变化
∂Z：Z 的无穷小变化
∂L/∂Z：敏感程度（有数值）
```
∂L/∂Z =
[
  [∂L/∂z11, ∂L/∂z12, ∂L/∂z13],
  [∂L/∂z21, ∂L/∂z22, ∂L/∂z23],
  [∂L/∂z31, ∂L/∂z32, ∂L/∂z33]
]
```
1. 这里需要理解导数, h是x变化量
f'(x) = lim_{h→0} [f(x+h) − f(x)] / h
2. 举例一元二次方程求导
f(x) = ax² + bx + c
f(x+h) = a(x+h)² + b(x+h) + c
f(x+h) − f(x) = 2axh + ah² + bh
[f(x+h) − f(x)] / h = 2ax + ah + b
取极限 h→0：f'(x) = 2ax + b
#### 从 ∂L/∂Z 到 Attention Weight
我们要计算：“W 中某一个格子 W[ia]，对 L 有多大影响？” 这就是 ∂L/∂W[ia]。
W[ia] 会影响 Z 的第 i 行的所有列 j
W[ia] → Z[i1], Z[i2], Z[i3], ..., Z[im]
每个 Z[i][j] 又会影响 L， 影响大小是 ∂L/∂Z[i][j]
W[ia] 对 L 的总影响，有 多条路径
W[ia]
 ├─→ Z[i1] → L
 ├─→ Z[i2] → L
 ├─→ Z[i3] → L
 └─→ ...

对于某一个 Z[i][j]：
W[ia] 对 Z[i][j] 的影响：∂Z[i][j] / ∂W[ia]
Z[i][j] 对 L 的影响： ∂L / ∂Z[i][j]
两者相乘，就是这条路径的贡献。
∂L/∂W[ia] = ∑_j ( ∂L/∂Z[i][j] ) × ( ∂Z[i][j]/∂W[ia] )

Z[i][j] = ∑_a W[i][a] × V[a][j]
W[i][a] 变化一点点，Z[i][j] 会变多少？ ∂Z[i][j] / ∂W[i][a] 
Z[i][j] = 
W[i][1]×V[1][j] +
W[i][2]×V[2][j] +
...
+ W[i][a]×V[a][j] +
...

∂Z[i][j]/∂W[ia] = V[a][j]
∂L/∂W[ia] = ∑_j ∂L/∂Z[i][j] × V[a][j]

#### 更新Weight
∂L/∂W = ∂L/∂Z @ Vᵀ
W.grad = ∂L/∂W
optimizer.step()

∂L/∂Scores = Weights ⊙ (∂L/∂Weights − mean(...))
∂L/∂Q, ∂L/∂K → ∂L/∂W_Q, ∂L/∂W_K

∑_j ∂L/∂Z[i][j] · ∂Z[i][j]/∂W[ia]
Z[i][j]需要清楚两维
W[ia]只是一个“被求导的元素”，所以没有写成W[i][a]


反向传播示例
∂L/∂Z
  ↓
∂L/∂Weights
  ↓ (softmax)
∂L/∂Scores
  ↓
∂L/∂Q, ∂L/∂K
  ↓
∂L/∂W_Q, ∂L/∂W_K, ∂L/∂W_V
  ↓
∂L/∂X


## token 基础
x = token_embedding + pos_embedding  # (B, T, D)
Q = x @ W_Q                          # (B, T, D_q) 
> “Q 是 token 的 query”
Q 是 token 的 query 向量表示
每个 token 对应一组 Q、K、V 向量
### FFN
FFN（Feed-Forward Network，前馈网络）是 Transformer 中每一个 Encoder / Decoder Layer 里的第二个核心组件（第一个是 Self-Attention）。
一句话说就是：
> FFN 是对每个 token 的特征向量做“独立非线性变换”的网络。
Input
 ↓
Self-Attention (+ Add & Norm)
 ↓
FFN (+ Add & Norm)
 ↓
Output
* Self-Attention：让 token 之间交换信息（谁和谁有关）。
* FFN：让每个 token 独立地做非线性变换（把当前信息映射成更有用的表示）。
 FFN典型做法是：先把维度放大（例如 512 维变成 2048 维），经过一个非线性激活（如 ReLU 或 GELU），再把维度缩小回原来的大小（2048 变回 512）。
所以常说 FFN 是“升维 → 非线性 → 降维”的两层全连接结构。
如果没有 FFN，模型容易变成本质上还是线性组合，难以学到更复杂的特征；
加上 FFN 后，每层才能让 token 的表示变得更“丰富”。
FFN 就是每个 token 独立过一个小 MLP，用来把当前信息映射成更有用的表示。
```python
def FFN(x):
    # x: [T, D]
    h = gelu(x @ W1 + b1)   # [T, D_ff]
    y = h @ W2 + b2        # [T, D]
    return y
```
```
x      : [T, D]
W1     : [D, D_ff]
b1     : [D_ff]
h      : [T, D_ff]
W2     : [D_ff, D]
b2     : [D]
y      : [T, D]
```
### Self-Attention 实战案例
1. 输入 “猫 喜欢 鱼”
2. 经过 tokenizer 后得到 3 个 token：`token_id = [23, 56, 78]`
3. embedding 4维
```python
x1 = [0.1, 0.2, 0.3, 0.4] # 猫
x2 = [0.5, 0.6, 0.7, 0.8] # 喜欢
x3 = [0.9, 1.0, 1.1, 1.2] # 鱼

X = # [T, D]
[
  [0.1, 0.2, 0.3, 0.4],   # 猫
  [0.5, 0.6, 0.7, 0.8],   # 喜欢
  [0.9, 1.0, 1.1, 1.2]    # 鱼
]
```
4. 计算 Q、K、V
Q = X @ WQ # 这个 token 发出的查询
K = X @ WK # 这个 token 被查询时的身份
V = X @ WV
n_heads = 2
D_head = D / n_heads = 2
W_V : [D, D_head]
```python
W_Q = # W_Q 是一个矩阵：[D, D_head]
[
  [1, 0],
  [0, 1],
  [1, 0],
  [0, 1]
]
W_K = # W_K：[D, D_head]
[
  [1, 0],
  [1, 0],
  [0, 1],
  [0, 1]
]
W_V =
[
  [1, 0],
  [0, 1],
  [0, 0],
  [1, 0]
]
```
得到
```python
Q = # [T, D_head]
[
  [0.1, 0.2],
  [0.5, 0.6],
  [0.9, 1.0]
]
K = # [T, D_head]
[ 
  [0.3, 0.4],
  [0.1, 0.5],
  [0.2, 0.3]
]
K^T = # [D_head, T]
[ 
  [0.3, 0.1, 0.2],
  [0.4, 0.5, 0.3],
]
V = 同上
```
5. 计算注意力分数（Attention Scores）
scores = Q @ K^T
得到
```python
scores =
[
  [0.05, 0.17, 0.29],
  [0.17, 0.61, 1.05],
  [0.29, 1.05, 1.81]
]
```
解释：
scores[i][j] = 第 i 个 token 对第 j 个 token 的“关注程度”
例如：
scores[1][2] = 1.05
表示 “喜欢” 对 “鱼” 关注度很高
6. Softmax（归一化成概率）
特点：
每一行加起来 = 1
越相关的位置越大
```python
attention_weights = # [T, T]
[ 
  [0.30, 0.33, 0.37],
  [0.15, 0.35, 0.50],
  [0.10, 0.30, 0.60]
]
```
7. 加权求和 V（得到输出）
output = attention_weights @ V
```python
output =
[
  [0.54, 0.64],   # 猫的新表示
  [0.62, 0.72],   # 喜欢的新表示
  [0.70, 0.80]    # 鱼的新表示
]
```
### 推理过程实战
因为 W_Q / W_K / W_V 是“参数”，不是“结果”；
```bash
W_Q ∈ ℝ^{d_model × d_q}
W_K ∈ ℝ^{d_model × d_k}
W_V ∈ ℝ^{d_model × d_v}
```
✅ 训练完就固定
✅ 不随输入变化
✅ 存的是这些矩阵的数值
```bash
Q = X @ W_Q
K = X @ W_K
V = X @ W_V
```
推理时输入的 token 不同，token对应X算出的`Q/K/V` 也不同。

## KV Cache
K_cache = [K₀, K₁, K₂, ..., Kₜ]
V_cache = [V₀, V₁, V₂, ..., Vₜ]
KV Cache 加速推理，是因为：
✅ 历史 token 的 K/V 不变
✅ 算一次，存起来复用
✅ 把每步 O(T) → O(1)（K/V 部分）
✅ 总复杂度从 O(T²) → O(T)

训练时：一次性算整个序列（不用 cache）
推理时：
每步只来一个新 token
新 token 的 K/V 单独算
历史 K/V 直接拼接
📌 缓存的是：
“已经算好的那一部分矩阵”

### TurboQuant
**解量化不一定回退到原值，但 TurboQuant 的设计目标就是：“误差足够小，生成质量几乎不变。”**
1. 假设真实值
`K = [0.11, 0.16, 0.21]`
2. INT4 映射：
找每行的 min / max，映射到 INT4（0~15）
`int4 = round((x - min)/(max - min) * 15)`
```bash
0.11 → 0
0.16 → ?
0.21 → 15
```
3. 计算 0.16：
`(0.16 - 0.11)/(0.21-0.11) = 0.5 → 0.5 × 15 = 7.5 → round → 8`
4. 解量化
`8 / 15 × 0.1 + 0.11 ≈ 0.163`
5. 为什么一定会有误差？ 
  1. 因为有限离散级别 INT4：只有 16 个值
  2. 范围压缩 所有值压进 [min, max]
  3. 四舍五入不可逆
6. 因此，量化 = 有损压缩，解量化 = 重建近似