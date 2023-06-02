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
    # nlp = PreProcessor()
    # analyzer = SentimentAnalyzer(nlp, use_openai_emb_api=False)

    # texts_train, labels_train = analyzer.read_data('dataset/train3098itemPOLARITY.csv',
    #                                                delimiter=';', text_index=2, label_index=1)
    # texts_test, labels_test = analyzer.read_data('dataset/test1326itemPOLARITY.csv',
    #                                              delimiter=';', text_index=2, label_index=1)
    # # texts_train_augmented = []
    # # labels_train_augmented = []
    # # augmenter = EmbeddingAugmenter(transformations_per_example=2)
    # # index = 0
    # # print('start augment')
    # # for text, label in zip(texts_train, labels_train):
    # #     text_aug_res = augmenter.augment(text)
    # #     label_aug_res = [label] * len(text_aug_res)
    # #     texts_train_augmented.extend(text_aug_res)
    # #     labels_train_augmented.extend(label_aug_res)
    # #     print(f'finish augment text {index}')
    # #     index += 1
    # # print('finish augment')
    # # with open('dataset/train3098itemPOLARITY_augmented0.csv', mode='w', encoding='utf-8') as f:
    # #     for text, label in zip(texts_train_augmented, labels_train_augmented):
    # #         line = f'xx;{label};{text}\n'
    # #         f.write(line)
    # # f.close()

    # analyzer.train('GB', texts_train, labels_train)
    # analyzer.test('GB', texts_test, labels_test)

    # # analyzer.test_gpt(texts_test, labels_test)

    # # analyzer.test_fine_tune(texts_test, labels_test)

    # # analyzer.train('GB', texts_train, labels_train, embedder='text-embedding-ada-002')
    # # analyzer.test('GB', texts_test, labels_test, embedder='text-embedding-ada-002')

    # # texts_1, labels_1 = analyzer.read_data('dataset/se-appreview.txt')
    # # texts_2, labels_2 = analyzer.read_data('dataset/se-sof4423.txt')
    # # texts, labels = analyzer.gen_dataset([texts_1, texts_2], [labels_1, labels_2])
    # # texts_train, texts_test, labels_train, labels_test = analyzer.split_dataset(texts, labels)
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
