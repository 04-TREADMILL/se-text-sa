model:
- 该模型的Embedding操作使用sklearn库的TfidfVectorizer在训练集上完成，训练好的向量化模型已保存为embedding-2023-05-21-19-41-39.pkl（见当前目录）
- 该模型的分类器由sklearn库中的SGDClassifier完成，训练好的分类器已保存为RF-local-2023-05-21-19-41-39.pkl（见当前目录）
- 调用方法：先通过训练好的TfidfVectorizer模型将文本转换为向量，再将向量作为训练好的RandomForestClassifier的输入得到预测结果

train:
- 代码示意如下（详见analyzer.py中的TFIDFAnalyzer类的train方法）：
    classifier = RandomForestClassifier()
    vectorizer = TfidfVectorizer(tokenizer=self.preprocessor.tokenize_and_stem, sublinear_tf=True, max_df=0.5, stop_words=self.preprocessor.stop_words, min_df=3)
    text_emb_train = vectorizer.fit_transform(text_train)
    classifier.fit(text_emb_train, label_train)
    save_model(vectorizer)
    save_model(classifier)
- 其中，self.preprocessor.tokenize_and_stem和self.preprocessor.stop_words分别为借助nltk库实现的tokenizer和自构建的停用词表，详见preProcessor.py
- 由于训练由sklearn库直接完成，没有留下训练日志

test:
- 测试结果详见当前目录下result.txt