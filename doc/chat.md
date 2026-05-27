## 语料基础
NLP = Natural Language Processing（自然语言处理）。
语料： Corpus（单数） / Corpora（复数）

### 大模型（如 GPT / LLaMA / PaLM）本质
`P(next_token | context)`: 给定一个上下文（已出现的文字），模型在预测“下一个词最可能是什么”。
**模型能力不是“参数或语料二选一”，而是：语料提供知识与模式，参数通过训练将其压缩为可泛化的函数。**
### 用“中国的首都”举例
1. context: `中国的首都`
2. 模型计算↓：
```bash
P("北京" | "中国的首都") ≈ 0.92
P("上海" | "中国的首都") ≈ 0.03
P("是"   | "中国的首都") ≈ 0.02
···
```
3. 训练目标：
最大化 `P(correct_next_token | context)`， 让“真实下一个词”的概率尽量高
4. 推理时
得到分布 `P(next_token | context)`, 用策略选一个 token：
```bash
greedy
top-k
top-p
```
重复这个过程 → 生成文本
**推理时：模型每次只生成 1 个 token，生成多少 token 完全由停止条件（max_tokens / EOS / 规则）决定。**
### 模拟
```bash
Prompt: "中国的首都"
生成 token 数 = 6
输出:
"中国的首都 北京 是 中国 的 城市"
```
过程：
```bash
1 → 北京
2 → 是
3 → 中国
4 → 的
5 → 城市
6 → <eos>
```
### 语料准备的 5 个核心原则（非常重要）
✅ 多样性 不同主题、风格、句式
✅ 规模 越大越好
✅ 干净 乱码、无广告
✅ 去重 不重复刷权重
✅ 平衡 不偏科

### SFT 格式（Supervised Fine-Tuning Format）
“教模型怎么当助手的说明书格式”。
✅ SFT 格式 = 指令 + 输入 + 输出​
✅ 告诉模型：用户说什么，助手该怎么回
普通语料: `今天天气很好。`
SFT 语料:
```bash
用户：今天天气怎么样？
助手：希望今天是晴天。
```
## 神经网络
**LSTM 是“递归记忆”，TransformerEncoder 是“全局注意力”；新项目优先 Transformer，老项目或流数据才考虑 LSTM。**
**RNN 是基础，LSTM 是改进，TransformerEncoder 是下一代；新项目默认 Transformer，流式或资源受限才用 LSTM，RNN 基本淘汰。**
LSTM（Long Short-Term Memory）
RNN（Recurrent Neural Network）
```bash
RNN
x₀ → h₀ → h₁ → h₂ → ...

LSTM
x₀ → (cell) → h₀ → (cell) → h₁ → ...

TransformerEncoder
x₀  ─┐
x₁  ─┼─ Attention ─►
x₂  ─┘
```

## ChatGPT 用了哪些语料？（公开已知）
1️⃣ 通用网页数据（最大头）
Common Crawl：全网爬取，PB 级
WebText / OpenWebText：Reddit 等高赞网页
Wikipedia：多语言百科
👉 占比：80% 以上
2️⃣ 书籍与长文本
BooksCorpus
Project Gutenberg
各类电子书、出版物
👉 用来学：长程依赖、逻辑、叙事
3️⃣ 代码数据
GitHub 公开代码
Stack Overflow
👉 用来学：逻辑推理、结构、因果
对话与指令数据（关键）
InstructGPT / RLHF 数据
人类标注的对话
人工排序（哪个回答更好）
👉 这部分才是 “让它像助手”的关键
