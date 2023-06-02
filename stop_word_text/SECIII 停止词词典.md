# SECIII 停止词词典

在本次实验的开始，我们选择了启发式规则+词典的形式，尝试通过对不同模型的改进和对数据进行预处理提高模型的预测准确性。停止词词典是提高准确度的重要手段。

我们寻找了常见的停止词词典，包括但不限于nltk自带的停止词、复现论文中的默认停止词典、stopwordsiso、smartstopwordslist、Kaggle中的停止词典和我们整合的停止词典，并通过脚本对整合的停止词典进行增删从而获得最好的性能。并将各词典应用于不同模型，记录其准确度并筛选最优结果。

可惜的是，在实验的后半程，我们决定选用的、效果最好的模型并没有使用停止词典，因此对停止词的选用并没有真正用于最后的模型。

下面是对采用的所有停止词典与classifier的结合中具有代表性的结果展示

| Stopwords + Classifier | rating |
| ---------------------- | ------ |
| nltk_stopwords + RF    | 82.21% |
| rank_nl_stopwords + RF | 83.21% |
| Smart_stopwords + RF   | 81.79% |
| spacy_stopwords + RF   | 81.74% |
| stopwordsiso + RF      | 80.74% |
| kaggle_stopwords + RF  | 82.16% |

