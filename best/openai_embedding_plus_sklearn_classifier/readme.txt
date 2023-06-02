model:
- 该模型的Embedding操作借助OpenAI的Embedding服务获取，选择了text-embedding-ada-002模型（可由任何有效的OpenAI API_KEY访问）
- 该模型的分类器由sklearn库中的SGDClassifier完成，训练好的分类器已保存为SGD-text-embedding-ada-002-2023-05-21-19-54-23.pkl（见当前目录）
- 调用方法：先通过OpenAI API_KEY调用text-embedding-ada-002模型将文本转换为向量，再将向量作为训练好的SGDClassifier的输入得到预测结果

train:
- 代码示意如下（详见analyzer.py中的OpenAIEmbedderAnalyzer类的train方法）：
    classifier = SGDClassifier()
    text_emb_train = get_embeddings(text_train)
    classifier.fit(text_emb_train, label_train)
    save_model(classifier)
- 由于训练由sklearn库直接完成，没有留下训练日志

test:
- 测试结果详见当前目录下result.txt