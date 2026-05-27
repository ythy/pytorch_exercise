### 错误记录
1. RuntimeError: stack expects each tensor to be equal size, but got [122] at entry 0 and [121] at entry 1
原因： dataset中 mask 必须和 x / y 等长

### 通用
1. 当前是LM模型 = 语言模型 = 预测下一个词
2. 100K语料预训练不到预期目标，改用500k语料尝试


### 预训练目标
loss 2.0 ~ 2.3​
生成 不重复
标点 正常
句子 能结束
### 过拟合
用`AdamW`替代`Adam`
AdamW: 把 Weight Decay 从梯度里“拆出来”，单独作用在参数上
```bash
# Adam（旧）
update = momentum_term + weight_decay * param

# AdamW（正确）
update = momentum_term
param = param - lr * weight_decay * param
```
### token算法
中文 BPE / SentencePiece：token数 ≈ 汉字数 × 0.7
情况 token / 字
单字为主 1.0
有词合并（天气、学习…）0.6～0.8
平均 ≈ 0.7