import os
import warnings

import nltk
import openai

from analyzer import TFIDFAnalyzer, ChatGPTAnalyzer, OpenAIEmbedderAnalyzer, OpenAIEnd2EndAnalyzer

warnings.filterwarnings("ignore")

openai.api_key = os.getenv('OPENAI_API_KEY')

nltk.download('universal_tagset', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

os.makedirs('./model/', exist_ok=True)


def main():
    analyzer1 = TFIDFAnalyzer()
    analyzer2 = OpenAIEmbedderAnalyzer()
    analyzer3 = ChatGPTAnalyzer()
    analyzer4 = OpenAIEnd2EndAnalyzer('ada:ft-personal-2023-05-30-05-21-25')

    analyzer1.analyze(
        train_file='dataset/train3098itemPOLARITY.csv',
        test_file='dataset/test1326itemPOLARITY.csv', algo='GB')

    analyzer2.analyze(
        train_file='dataset/train3098itemPOLARITY.csv',
        test_file='dataset/test1326itemPOLARITY.csv', algo='GB')

    analyzer3.analyze(test_file='dataset/test1326itemPOLARITY.csv')

    analyzer4.analyze(test_file='dataset/test1326itemPOLARITY.csv')


if __name__ == '__main__':
    main()
