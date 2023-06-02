# SE-Text-Sentiment-Analysis

针对软件工程文本进行情绪分析，具体表现为使用分类模型将文本分类为 `positive`、`negative` 和 `neutral`

代码框架和基本处理思路来自于：https://github.com/senticr/SentiCR

#### 处理思路

- 对文本数据进行预处理
- 将预处理后的文本数据嵌入成向量
- 将数据集划分成训练集和测试集
- 使用训练集训练一个三分类器
- 将训练好的模型在测试集上进行测试

#### 可以推进的点

- 方向一：预处理，尝试不同的文本预处理方法
- 方向二：文本嵌入，尝试不同的嵌入方法或模型
- 方向三：分类器，尝试不同的分类器算法

#### 基于 SentiCR

- 使用 NLP 算法，进行展开缩写、去除 URL 等预处理（方向一）
- 使用 TF-IDF 的方法将文本转化为向量（方向二）
- 使用 sklearn 库中不同的分类器进行训练和测试（方向三）

#### 个人尝试

- 使用 OpenAI 的 embedding 模型将文本嵌入成向量（需要用到 OpenAI API）（方向二）
- 将新的文本嵌入方法与 sklearn 库中不同的分类器组合进行训练和测试（方向三）

#### 新思路

- 通过 Prompt 的方式，使用 ChatGPT 端到端地完成分类
- 使用训练集对 GPT3 模型进行微调，端到端地完成分类

#### 实验方式

对不同的文本向量化方法，分别训练一系列不同的分类器，比较预测效果

注：由于经过 SentiCR 的预处理后的文本在使用 OpenAI 的 embedding 模型进行向量嵌入时遇到了报错，并且考虑到一条文本通过该模型被嵌入到对应向量空间时一定程度上自带有语义信息，所以在使用 OpenAI 的 embedding 模型时没有进行文本预处理步骤

#### 实验结果

**实验一**：

- 使用迭代二中助教提供的软工文本数据集 `sof4423` 和 `app-review`，将两者合并打乱并按照 6:4 的比例划分为训练集和测试集
- 文本向量化方法选用 TF-IDF 方法和 OpenAI 的 text-embedding-ada-002 模型，分类算法选择 sklearn 库中可以调用的一系列机器学习分类器（LinearSVC, BernoulliNB, SGDClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, MLPClassifier），两两组合记录预测结果
- 实验数据：（详细数据见 `./results/0.txt`）

| 排名 | 文本向量化方法         | 分类器算法                 | 预测准确率 |
| ---- | ---------------------- | -------------------------- | ---------- |
| 1    | text-embedding-ada-002 | LinearSVC                  | **85.52%** |
| 2    | text-embedding-ada-002 | SGDClassifier              | **85.20%** |
| 3    | TF-IDF                 | GradientBoostingClassifier | **82.00%** |
| 4    | TF-IDF                 | RandomForestClassifier     | 81.85%     |
| 5    | text-embedding-ada-002 | MLPClassifier              | 81.69%     |
| 6    | text-embedding-ada-002 | GradientBoostingClassifier | 79.80%     |
| 7    | TF-IDF                 | LinearSVC                  | 79.22%     |
| 8    | TF-IDF                 | SGDClassifier              | 78.17%     |
| 9    | TF-IDF                 | AdaBoostClassifier         | 78.12%     |
| 10   | text-embedding-ada-002 | RandomForestClassifier     | 77.96%     |
| 11   | TF-IDF                 | MLPClassifier              | 76.76%     |
| 12   | text-embedding-ada-002 | BernoulliNB                | 76.71%     |
| 13   | TF-IDF                 | DecisionTreeClassifier     | 74.92%     |
| 14   | text-embedding-ada-002 | AdaBoostClassifier         | 72.51%     |
| 15   | TF-IDF                 | BernoulliNB                | 71.09%     |
| 16   | text-embedding-ada-002 | DecisionTreeClassifier     | 58.03%     |

**实验二**：

- 使用助教指定的训练集和测试集（由数据集 `sof4423` 预划分而来）
- 文本向量化方法选用 TF-IDF 方法和 OpenAI 的 text-embedding-ada-002 模型，分类算法选择 sklearn 库中可以调用的一系列机器学习分类器（LinearSVC, BernoulliNB, SGDClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier, MLPClassifier），两两组合记录预测结果
- 实验数据：（详细数据见 `./results/1.txt`）

| 排名 | 文本向量化方法         | 分类器算法                 | 预测准确率 |
| ---- | ---------------------- | -------------------------- | ---------- |
| 1    | text-embedding-ada-002 | SGDClassifier              | **86.43%** |
| 2    | text-embedding-ada-002 | LinearSVC                  | **85.67%** |
| 3    | TF-IDF                 | RandomForestClassifier     | **83.79%** |
| 4    | TF-IDF                 | GradientBoostingClassifier | **83.33%** |
| 5    | TF-IDF                 | SGDClassifier              | **82.96%** |
| 6    | TF-IDF                 | LinearSVC                  | **82.88%** |
| 7    | text-embedding-ada-002 | MLPClassifier              | **82.28%** |
| 8    | TF-IDF                 | MLPClassifier              | 81.98%     |
| 9    | TF-IDF                 | AdaBoostClassifier         | 81.00%     |
| 10   | text-embedding-ada-002 | GradientBoostingClassifier | 80.24%     |
| 11   | TF-IDF                 | DecisionTreeClassifier     | 78.81%     |
| 12   | text-embedding-ada-002 | RandomForestClassifier     | 76.40%     |
| 13   | text-embedding-ada-002 | BernoulliNB                | 76.09%     |
| 14   | TF-IDF                 | BernoulliNB                | 75.94%     |
| 15   | text-embedding-ada-002 | AdaBoostClassifier         | 71.95%     |
| 16   | text-embedding-ada-002 | DecisionTreeClassifier     | 56.49%     |

**实验三**：

- 使用助教指定的训练集和测试集（由数据集 `sof4423` 预划分而来）
- 通过 Prompt 的方式，直接使用 ChatGPT（gpt-3.5-turbo）完成端到端的软工文本情绪分类任务（需要用到 OpenAI API）

~~~python
import json
import openai

text = '<text under test>'
prompt = f"""
Identify the emotion of the software engineering text delimited by triple backticks.
 ```{text}```
Classify it as 'positive' or 'negative' or 'neutral'.
Provide answer in JSON format with key 'label', which is the classification result.
"""
messages = [{'role': 'user', 'content': prompt}]
response = openai.ChatCompletion.create(
	model='gpt-3.5-turbo',
	messages=messages,
	temperature=0,
)
result = response.choices[0].message['content']
result = result[result.find('{'):result.rfind('}') + 1]
y_pred = json.loads(result)['label']
~~~

- 实验数据：（详细数据见 `./results/1.txt`）

初步尝试预测准确率不够理想，仅有 **74.66%**，由于 API 网络通信和 OpenAI 相关服务器负载问题，测试过程经常被迫中止，测试时间开销极大，因此放弃了通过优化 Prompt 方式来提高预测准确率的想法

**实验四**：

- 使用助教指定的训练集和测试集（由数据集 `sof4423` 预划分而来）
- 使用训练集对 OpenAI 的 Ada 模型进行微调，以完成端到端的软工文本情绪分类任务，在测试集上进行测试

将用于微调的数据集（训练集）转换为 JSONL 格式，每条数据的表示方式如下：

```json
{"prompt": "<prompt text>", "completion": "<ideal generated text>"}
```

使用 OpenAI CLI 微调基础模型，这里选用了参数量最少且响应速度最快的 Ada 模型：

```
openai api fine_tunes.create -t <TRAIN_FILE_ID_OR_PATH> -m ada
```

通过以下命令检测微调任务进度

```
openai api fine_tunes.follow -i <YOUR_FINE_TUNE_JOB_ID>
```

微调完成后可以获得新模型的名称，在 Python 代码中调用：

```python
import openai

text = '<text under test>'
fine_tune_model = '<fine_tuned_model_name>'

response = openai.Completion.create(
	model=fine_tune_model,
	prompt=text,
	temperature=0,  # 控制模型回答内容的创造性, 取值为[0,2], 对分类任务而言设定为0 
	max_tokens=1,  # 限制模型回答的Token数量, 否则模型的回答大概率包含多余的内容
)
y_pred = response['choices'][0]['text']
```

- 实验数据：（详细数据见 `./results/1.txt`）

初次尝试得到的预测准确率为 **86.35%**，与实验二中表现最好的方法效果相当，但观察测试结果发现，模型对某些文本给出的回答不在 `positive`、`negative` 和 `neutral` 这三个单词的范围内，若将不是 `positive` 或 `negative` 的回答全部算作 `neutral`，对实验结果没有产生实质影响

为了进一步挖掘大模型的能力，选择对训练集和测试集进行特殊处理，在每一段文本的末尾添加字符串 `" ->"` （提示模型需要进行情感分类的软工文本到此已经结束，接下来需要回答分类结果了），再次进行模型微调得到新模型，预测准确率有了显著提升，达到 **90.05%**

**实验五：**

- 使用助教指定的训练集和测试集（由数据集 `sof4423` 预划分而来），TF-IDF进行文本向量化，GradientBoostingClassifier分类器，与不同的预处理算法进行组合进行实验
- SentiCR中原有的预处理方法有：word stem，对表情符号，缩写，否定的处理，去除了url，md图片，html标签
- 我们新加的预处理方法有：
  - 将文本中的专有词和实体词用标签替代
  - 处理斜体和大写单词
  - 处理在软工领域有特别含义的单词，如：support, bug, error
- 实验数据：（详细数据见./result/2.txt）

| 预处理方法                               | 预测准确率    |
| ---------------------------------------- | ------------- |
| 原有方法                                 | 82.50%/82.88% |
| 原有方法+替换专有词和实体词              | 81.90%        |
| 原有方法+处理领域特定单词                | 82.65%        |
| 原有方法+处理斜体                        | 82.43%        |
| 原有方法+处理大写                        | 82.58%        |
| 原有方法+处理斜体、大写                  | 82.80%/82.58% |
| 原有方法+处理斜体、大写+处理领域特定单词 | 82.65%/82.58% |

- 实验结论：在TF-IDF和GradientBoostingClassifier的条件下，与处理算法对于结果的影响不大。
