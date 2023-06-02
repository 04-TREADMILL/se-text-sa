import csv
import json
import os
import pickle
from datetime import datetime
from typing import Tuple, List

import numpy as np
import openai
from openai.embeddings_utils import get_embeddings
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from preProcessing import PreProcessor


class Analyzer:
    def __init__(self):
        self.preprocessor = PreProcessor()

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def analyze(self, *args, **kwargs):
        pass

    @staticmethod
    def print_result(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Print and return the results of a classification model.

        :param y_true: The true labels as a NumPy array.
        :param y_pred: The predicted labels as a NumPy array.
        :return: precision, recall, F1 score, and accuracy.
        """
        if y_true.shape != y_pred.shape:
            print('Invalid parameters. Shape of `y_true` and `y_pred` does not match.')
            return
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1score = f1_score(y_true, y_pred, average=None)
        accuracy = accuracy_score(y_true, y_pred)
        print('testing result:')
        print(f'- Precision: {precision}')
        print(f'- Recall:    {recall}')
        print(f'- F-measure: {f1score}')
        print(f'- Accuracy:  {accuracy}')
        print(f'- Classification Report:\n{classification_report(y_true, y_pred, digits=4)}')
        return precision, recall, f1score, accuracy

    @staticmethod
    def read_data(file_name: str, delimiter=';', text_index=2, label_index=1) -> Tuple[List[str], List[str]]:
        """
        Read data from a file and extract text and label information based on provided indices.

        :param file_name: The name (and path if necessary) of the file to be read.
        :param delimiter: parameter for delimiter in csv.reader()
        :param text_index: The index of the text in each line of the file. Defaults to 1.
        :param label_index: The index of the label in each line of the file. Defaults to 2.
        :return: A list of texts and a list of corresponding labels.
        """
        with open(file_name, encoding='utf-8', mode='r') as file:
            file_reader = csv.reader(file, delimiter=delimiter)
            dataset = [(line[text_index], line[label_index]) for line in file_reader]
            texts = [data[0] for data in dataset]
            labels = [data[1] for data in dataset]
        file.close()
        return texts, labels

    @staticmethod
    def split_dataset(texts: List[str], labels: List[str], test_size=0.4, seed=42) \
            -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Split the dataset into training and testing sets.

        :param texts: A list of texts.
        :param labels: A list of labels corresponding to the texts.
        :param test_size: The proportion of the dataset to include in the test split. Default value is 0.4.
        :param seed: Random seed to ensure reproducibility. Default value is 42.
        :return: Four lists: the first one is a list of texts for training, the second one is a list of texts
        for testing, the third one is a list of labels for training and the fourth one is a list of labels for testing.
        """
        texts = np.array(texts)
        labels = np.array(labels)
        texts_train, texts_test, labels_train, labels_test \
            = train_test_split(texts, labels, test_size=test_size, random_state=seed)
        return texts_train.tolist(), texts_test.tolist(), labels_train.tolist(), labels_test.tolist()


class LocalAnalyzer(Analyzer):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.algos = {'SVM': LinearSVC(),
                      'NB': BernoulliNB(),
                      'SGD': SGDClassifier(),
                      'AB': AdaBoostClassifier(),
                      'RF': RandomForestClassifier(),
                      'GB': GradientBoostingClassifier(),
                      'DT': DecisionTreeClassifier(),
                      'MLP': MLPClassifier(activation='logistic', batch_size='auto',
                                           early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
                                           learning_rate_init=0.1, max_iter=5000, random_state=1,
                                           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                                           warm_start=False)}

    def get_embeddings(self, *args, **kwargs) -> np.ndarray:
        pass

    def load_model(self, *args, **kwargs) -> bool:
        pass


class TFIDFAnalyzer(LocalAnalyzer):
    def __init__(self):
        super().__init__()
        self.vectorizer = None

    def get_embeddings(self, texts, mode='test'):
        if mode == 'train':
            self.vectorizer = TfidfVectorizer(tokenizer=self.preprocessor.tokenize_and_stem, sublinear_tf=True,
                                              max_df=0.5, stop_words=self.preprocessor.stop_words, min_df=3)
            return self.vectorizer.fit_transform(texts).toarray()

        assert self.vectorizer is not None
        return self.vectorizer.transform(texts).toarray()

    def load_model(self, algo) -> bool:
        all_files = os.listdir('./model/')
        classifier_models = sorted([f for f in all_files if algo in f and 'local' in f], reverse=True)
        embedding_models = sorted([f for f in all_files if 'embedding' in f], reverse=True)
        classifier_model = classifier_models[0] if len(classifier_models) > 0 else None
        embedding_models = [model for model in embedding_models if model[model.index('-'):] in classifier_model] \
            if classifier_model is not None else []
        embedding_model = embedding_models[0] if len(embedding_models) > 0 else None
        if classifier_model is not None and embedding_model is not None:
            with open(f'./model/{classifier_model}', 'rb') as f1, \
                    open(f'./model/{embedding_model}', 'rb') as f2:
                self.models[algo] = pickle.load(f1)
                self.vectorizer = pickle.load(f2)
            print(f'- Model loaded: {classifier_model}')
            print(f'- Model loaded: {embedding_model}')
            return True
        return False

    def train(self, algo: str, x_train: List[str], y_train: List[str]):
        model = self.algos[algo]
        x_train = [self.preprocessor.preprocess_text(text) for text in x_train]
        x_train = self.get_embeddings(x_train, mode='train')
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        model.fit(x_train, y_train)
        curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_name = f'{algo}-local-{curr_time}.pkl'
        with open(f'./model/{model_name}', 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f'- Model saved: {model_name}')
        model_name = f'embedding-{curr_time}.pkl'
        with open(f'./model/{model_name}', 'wb') as emb_file:
            pickle.dump(self.vectorizer, emb_file)
        print(f'- Model saved: {model_name}')

    def test(self, algo: str, texts: List[str], y_true: List[str]):
        if not self.load_model(algo):
            print('Fail to load model!')
            return None, None, None, None

        classifier = self.models[algo]
        texts = [self.preprocessor.preprocess_text(text) for text in texts]
        embeddings = self.get_embeddings(texts)
        y_pred = classifier.predict(embeddings)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return self.print_result(y_true, y_pred)

    def analyze(self, train_file, test_file, algo):
        texts_train, labels_train = self.read_data(train_file)
        texts_test, labels_test = self.read_data(test_file)
        self.train(algo, texts_train, labels_train)
        self.test(algo, texts_test, labels_test)


class OpenAIEmbedderAnalyzer(LocalAnalyzer):
    def __init__(self):
        super().__init__()
        self.optional_apis = ['text-embedding-ada-002',
                              'text-similarity-ada-001',
                              'text-similarity-babbage-001',
                              'text-similarity-curie-001',
                              'text-similarity-davinci-001']

    def get_embeddings(self, texts, embedder, mode='test') -> np.ndarray:
        """
        Generate the embeddings for a list of texts using a specified embedder.

        :param texts: The list of texts to be processed and embedded.
        :param embedder: The method used for generating the embeddings.
        :param mode: The mode of operation. If 'train' is specified, the vectorizer is fitted on the provided texts.
        :return: The generated embeddings for the given list of texts.
        """
        assert embedder in self.optional_apis
        MAX_LIMIT = 2048
        if len(texts) <= MAX_LIMIT:
            return np.array(get_embeddings(list_of_text=texts, engine=embedder))

        embeddings = []
        start = 0
        end = MAX_LIMIT
        while len(texts) >= end != start:
            embeds = get_embeddings(list_of_text=texts[start:end], engine=embedder)
            embeddings.extend(embeds)
            start = end
            end = min(start + MAX_LIMIT, len(texts))
        return np.array(embeddings)

    def __load_model(self, algo, embedder) -> bool:
        assert embedder in self.optional_apis
        all_files = os.listdir('./model/')
        classifier_models = sorted([f for f in all_files if algo in f and embedder in f], reverse=True)
        classifier_model = classifier_models[0] if len(classifier_models) > 0 else None
        if classifier_model is not None:
            with open(f'./model/{classifier_model}', 'rb') as f:
                self.models[algo] = pickle.load(f)
            print(f'- Model loaded: {classifier_model}')
            return True
        return False

    def train(self, algo: str, x_train: List[str], y_train: List[str], embedder: str = ''):
        if embedder not in self.optional_apis:
            print('Invalid openai embedding api name!')
            print('You can choose from the following api names:')
            for api in self.optional_apis:
                print(f'- {api}')
            return

        model = self.algos[algo]
        x_train = [self.preprocessor.preprocess_text_v2(text) for text in x_train]
        x_train = self.get_embeddings(x_train, embedder, mode='train')
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        model.fit(x_train, y_train)
        curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_name = f'{algo}-{embedder}-{curr_time}.pkl'
        with open(f'./model/{model_name}', 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f'- Model saved: {model_name}')

    def test(self, algo: str, texts: List[str], y_true: List[str], embedder: str = ''):
        if not self.load_model(algo, embedder):
            print('Fail to load model!')
            return None, None, None, None

        classifier = self.models[algo]
        texts = [self.preprocessor.preprocess_text_v2(text) for text in texts]
        embeddings = self.get_embeddings(texts, embedder)
        y_pred = classifier.predict(embeddings)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return self.print_result(y_true, y_pred)

    def analyze(self, train_file, test_file, algo):
        texts_train, labels_train = self.read_data(train_file)
        texts_test, labels_test = self.read_data(test_file)
        self.train(algo, texts_train, labels_train)
        self.test(algo, texts_test, labels_test)


class OpenAIEnd2EndAnalyzer(Analyzer):
    def __init__(self, model_name):
        super().__init__()
        self.fine_tuned_model = model_name

    def train(self):
        print('Please create your fine-tuned model via OpenAI CLI.')
        print('Usage: openai api fine_tunes.create -t <FILE_PATH> -m <MODEL_NAME>')

    def test(self, texts: List[str], y_true: List[str], special_prompt=' ->'):
        texts = [self.preprocessor.preprocess_text_v2(text) for text in texts]
        y_pred = []
        for i in range(len(texts)):
            text = texts[i]
            print(f'Testing {i}')
            response = openai.Completion.create(
                model=self.fine_tuned_model,
                prompt=text + special_prompt,
                temperature=0,
                max_tokens=1,
            )
            result = response['choices'][0]['text']
            if 'negative' in result:
                y_pred.append('negative')
            elif 'positive' in result:
                y_pred.append('positive')
            else:
                y_pred.append('neutral')
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return self.print_result(y_true, y_pred)

    def analyze(self, test_file):
        texts_test, labels_test = self.read_data(test_file)
        self.test(texts_test, labels_test)


class ChatGPTAnalyzer(Analyzer):
    def train(self):
        print("OpenAI has done this for you. Just use it properly.")

    def test(self, texts: List[str], y_true: List[str], start=0):
        y_pred = []
        for i in range(start, len(texts)):
            text = texts[i]
            prompt = f"""
            Identify the emotion of the software engineering text delimited by triple backticks.
            ```{text}```
            Classify it as 'positive' or 'negative' or 'neutral'.
            Provide answer in JSON format with key 'label', which is the classification result.
            """
            messages = [{'role': 'user', 'content': prompt}]
            print(f'Waiting for response from chatGPT ... {i}')
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=0
            )
            result = response.choices[0].message['content']
            result = result[result.find('{'):result.rfind('}') + 1]
            result = json.loads(result)
            y_pred.append(result['label'])
            print(str(i) + ': ' + str(y_pred))
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return self.print_result(y_true, y_pred)

    def analyze(self, test_file):
        texts_test, labels_test = self.read_data(test_file)
        self.test(texts_test, labels_test)
