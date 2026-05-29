# 随录
## GPT2 模型结构
```baSH
GPT2LMHeadModel
└── GPT2Model
    ├── wte        # token embedding
    ├── wpe        # position embedding
    ├── h (layers)
    │   ├── attn
    │   │   ├── c_attn.weight  ✅ wQ + wK + wV 拼在一起
    │   │   ├── c_proj.weight
    │   ├── mlp
    │   │   ├── c_fc.weight
    │   │   ├── c_proj.weight
    ├── ln_f
└── lm_head.weight  # 输出投影 lm_head = Linear(hidden_size, vocab_size)

```
## MLP = 前馈神经网络（Feed‑Forward Network）
```bash
transformer.h[layer].mlp
├── c_fc   (升维) # fully-connected layer
├── act    (激活) # activation function
└── c_proj (降维) # projection layer
```

## 当前百科数据适合什么
✅ 适合训练：因果语言模型（CLM）→ QA‑finetune / Pretrain + SFT​
✅ 推荐模型：GPT‑2（中文需 SentencePiece）/ Qwen‑0.5B / ChatGLM‑6B‑int4
### 方案一（最推荐·你项目最匹配）：GPT‑2 + QA‑SFT
```bash
<s>问：小华要冻些冰块来调酒，为了赶时间，她应该？</s>答：就用冷开水</s>
<s>问：小黄花鱼长大后就是大黄花鱼吗？</s>答：不是</s>
<s>问：世界上最大的蝴蝶是哪一种？</s>答：南美凤蝶</s>
<s>问：小鸡是由鸡蛋的？答：蛋黄发育来的</s>
```
### 方案二（工业级·省事）：Qwen‑0.5B / 1.5B + LoRA
```bash
{"instruction": "回答以下问题", "input": "小华要冻些冰块来调酒，为了赶时间，她应该？", "output": "就用冷开水"}
```

## ReLU（Rectified Linear Unit）
```py
def relu(x):
    return x if x > 0 else 0
```
## GeLU（Gaussian Error Linear Unit）
GELU 是一种 平滑、带“概率感”的**激活函数**，把输入映射成非线性输出. 也是 GPT / BERT / Transformer 的 核心非线性来源。
“以某个概率随机决定是否让 x 通过
```bash
x ≫ 0 几乎一定通过
x ≈ 0 半通
x < 0 小概率通过
```
激活函数 = 给神经网络引入“非线性”的部件,  没有它，多层 Transformer = 一层线性
```py
def forward(x):
    return 0.5 * x * (1.0 + torch.tanh(
        (2.0 / 3.1415926535) ** 0.5 *
        (x + 0.044715 * torch.pow(x, 3))
    ))
```

## SFT​
SFT:​ Supervised Fine‑Tuning（监督微调）
用「输入 → 正确输出」的数据继续训模型
## LoRA​
LoRA​: Low‑Rank Adaptation（低秩适配）
只训一小撮新增参数，不动原模型权重
对立的是 `Full Fine‑Tuning`
SFT 是“训练目标”，LoRA 是“训练方法”

## Causal LM 
Causal LM  = 自回归语言模型（Autoregressive LM）
每个 token 只能看到“它自己和之前的 token”
```bash
Transformer (抽象)
 └─ 输出 hidden_states
     └─ LM Head
         └─ Loss
```
## GPT2Model
GPT2Model是 「 Decoder‑only Transformer（自回归 Transformer）」
GPT2Model = Decoder-only
          = 自己看自己
1. 没有“先编码再解码”的过程↓
```
输入：0531 是哪个城市的区号？
↓（同一个模型）
输出：济南
```
2. 怎么“理解输入”？
靠 Decoder 自己的 Self‑Attention, 适合生成答案，Q/A, 不适合复杂推理
```bash
Token 0 → 学表示
Token 1 → 看 Token 0
Token 2 → 看 Token 0,1
...
```
3. LM Head = Transformer 输出 → vocab_size 的 Linear 层

## `torch._dynamo.disable()`
关闭 PyTorch 的 torch.compile / torch._dynamo 动态编译系统。否则报错
## 4440个词表 Loss = 1.5 意味什么
正确 token 概率：~0.25
错误 token 概率：~0.75 分散在 4399 个词上
模型“知道一点”，但不敢确定。 一个错，后面全歪
## 原生用法 -1 取list 最后一个索引
## ASK 时 temperature 的作用
结论：  
T < 1 = 增加高概率 token 的选中概率
T > 1 = 给冷门 token 机会
```bash
logits = [2.0, 1.0, 0.1] 
logits_softmax = [0.66, 0.24, 0.10] # softmax（T = 1）

logits / 0.5 = [4.0, 2.0, 0.2] #（T = 0.5）
logits_softmax =  [0.88, 0.12, 0.00] # softmax（T = 0.5）

logits / 2 = [1.0, 0.5, 0.05] #（T = 2.0）
logits_softmax =   [0.58, 0.35, 0.07] # softmax（T = 2.0）  
```
## ASK 时 Top‑K 作用 
Top‑K = 只保留概率最高的 K 个 token，其余全部屏蔽
```bash
if top_k > 0:
    k = min(top_k, next_logit.size(-1)) # 词表大小4400 top_k = 50  →  k = 50； top_k = 50000   →  k = 4400
    values, _ = torch.topk(next_logit, k) # values = [最高, 次高, ..., 第 K 高] # 取前 K 大的 logits
    next_logit[next_logit < values[-1]] = float("-inf")
```
## ASK 时 Top‑p 作用 
Top‑p = 保留 “累计概率 ≤ p 的最小 token 集合”
```bash
 if top_p < 1.0:
    sorted_logits, sorted_indices = torch.sort(next_logit, descending=True) #降序 sorted_logits next_logit 维度都是[4400]
    # [ 累计概率说明
    #     0.60,       # 第 1 个
    #     0.80,       # 第 1 + 2 个
    #     0.90,       # 第 1 + 2 + 3 个
    #     0.95,       # + 第 4 个
    #     0.98,       # + 第 5 个
    #     ...
    # ]
    cumulative_probs = torch.cumsum( # 累计概率
        torch.softmax(sorted_logits, dim=-1), dim=-1 # torch.softmax(sorted_logits, dim=-1) [4400] 按概率排序 
    )
    sorted_logits[cumulative_probs > top_p] = float("-inf") #  截断 > top_p
    next_logit.scatter_(0, sorted_indices, sorted_logits) # 通过sorted_indices，把next_logit中对应的token值变为-inf
```



## 分析 x,y 各是什么
```bash
def __getitem__(self, idx):
        with open(self.file, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
            q, a = line.strip().split("\t", 1)

        text = f"<s>{q}</s>{a}"
        ids = self.sp.Encode(text, out_type=int)[:self.max_len]
        return ids[:-1], ids[1:]

# line:  0531 是哪个城市的区号？	济南
# text = "<s>0531 是哪个城市的区号？</s>济南"
# ids = [1, 101, 102, 103, 104, 105, 106, 107, 2, 201, 202]
# ids[:-1] = [1, 101, 102, 103, 104, 105, 106, 107, 2, 201]
# ids[1:]   = [101, 102, 103, 104, 105, 106, 107, 2, 201, 202]
# x, y 分别对应2个向量
# x = [1,   101, 102, 103, 104, 105, 106, 107, 2,   201]
# y = [101, 102, 103, 104, 105, 106, 107, 2,   201, 202]

X = [[1, 101, 102, 103, 104, 105, 106, 107, 2, 201]]
Y = [[101, 102, 103, 104, 105, 106, 107, 2, 201, 202]]
# W_token: [4400, 256] 词嵌入矩阵	
# H = W_token[X] H.shape = [1, 10, 256]  → Hidden States（隐状态）是 Transformer 每一层输出的“语义向量”

# Logits = 模型在每个位置、对每个 token 的“打分” 越大 = 模型越认为该 token 应该出现
Logits.shape = [1, 10, 4400]
Logits.shape = [B, T, V]
               ↑  ↑  ↑
               │  │  └─ vocab_size (4400)
               │  └──── seq_len
               └─────── batch
# 第 j 个数 = 模型认为“下一个 token 是 j”的置信度
Logits[i] = [
   -2.31,   # token 0
    1.87,   # token 1
   -0.42,   # token 2
    ...
    3.91    # token 4399
]    
# Y[i]是正确答案， CE让 Logits[i, Y[i]] 尽可能大 让 Logits[i, others] 尽可能小
loss = CE( # CE = torch.nn.CrossEntropyLoss() 
    Logits.view(-1, 4400),  # [10, 4400] ← [B × T, 4400] ← [B, T, V]
    Y.view(-1)              # [10] ← [B, T]
)
```

# GPT 训练矩阵速查表

## 一、全局配置

| 参数 | 值 |
|---|---|
| batch_size | 2 |
| seq_len (T) | 10 |
| vocab_size (V) | 4400 |
| n_embd (D) | 256 |
| n_head | 4 |
| head_dim | 64 |

---

## 二、数据流总览

| 阶段 | 张量形状 |
|---|---|
| Input IDs | `[B, T]` |
| Token Embedding | `[B, T, D]` |
| Position Embedding | `[T, D]` |
| Transformer Input | `[B, T, D]` |
| Attention Output | `[B, T, D]` |
| MLP Output | `[B, T, D]` |
| Final Hidden States | `[B, T, D]` |
| Logits | `[B, T, V]` |
| Loss | `scalar` |

---

## 三、Embedding

| 操作 | 说明 | 形状 |
|---|---|---|
| `X` | token ids | `[B, T]` |
| `W_token` | 词嵌入矩阵 | `[V, D]` |
| `H_token` | 查表 | `[B, T, D]` |
| `W_pos` | 位置编码 | `[max_len, D]` |
| `H` | 合成表示 | `[B, T, D]` |

✅ **Embedding = lookup，不是矩阵乘法**

---

## 四、Attention

| 变量 | 形状 |
|---|---|
| Q, K, V | `[B, T, D]` |
| QKV | `[B, T, 3D]` |
| Q (多头) | `[B, n_head, T, head_dim]` |
| K (多头) | `[B, n_head, T, head_dim]` |
| V (多头) | `[B, n_head, T, head_dim]` |
| Attention Scores | `[B, n_head, T, T]` |
| Attention Output | `[B, T, D]` |

✅ **因果 mask 上三角为 −∞**

---

## 五、MLP   = Multilayer Perceptron Regression(回归), 模型类型
 