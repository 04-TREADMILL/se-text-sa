import csv
import os
import pickle
import re
import warnings
from datetime import datetime
from typing import List, Tuple

import nltk
import numpy as np
import openai

from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split

from openai.embeddings_utils import get_embeddings, get_embedding
from sklearn.feature_extraction.text import TfidfVectorizer
# from gensim.models import Word2Vec
# from gensim.scripts import glove2word2vec

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

openai.api_key = os.getenv('OPENAI_API_KEY')

nltk.download('universal_tagset', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)


class NLP:
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
        self.stop_words = [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ourselves', 'you', 'your',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'themselves',
            'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
            'and', 'if', 'or', 'as', 'until', 'of', 'at', 'by', 'between', 'into',
            'through', 'during', 'to', 'from', 'in', 'out', 'on', 'off', 'then', 'once', 'here',
            'there', 'all', 'any', 'both', 'each', 'few', 'more',
            'other', 'some', 'such', 'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
                                                                                                      'while', 'case',
            'switch', 'def', 'abstract', 'byte', 'continue', 'native', 'private',
            'synchronized', 'if', 'do', 'include', 'each', 'than', 'finally', 'class', 'double',
            'float', 'int', 'else', 'instanceof', 'long', 'super', 'import', 'short', 'default',
            'catch', 'try', 'new', 'final', 'extends', 'implements', 'public', 'protected', 'static',
            'this', 'return', 'char', 'const', 'break', 'boolean', 'bool', 'package', 'byte', 'assert',
            'raise', 'global', 'with', 'or', 'yield', 'in', 'out', 'except', 'and', 'enum', 'signed',
            'void', 'virtual', 'union', 'goto', 'var', 'function', 'require', 'print', 'echo', 'foreach',
            'elseif', 'namespace', 'delegate', 'event', 'override', 'struct', 'readonly', 'explicit',
            'interface', 'get', 'set', 'elif', 'for', 'throw', 'throws', 'lambda', 'endfor', 'endforeach',
            'endif', 'endwhile', 'clone'
        ]
        self.url_regex = re.compile(r'https?://(?:[a-zA-Z]|\d|[$-_@.&+]|[!*(),]|(?:%[\da-fA-F][\da-fA-F]))+')
        self.negation_words = ['not', 'never', 'none', 'nobody', 'nowhere', 'neither', 'barely', 'hardly', 'nothing',
                               'rarely', 'seldom', 'despite']
        self.emoticon_words = ['PositiveSentiment', 'NegativeSentiment']
        self.emoticon_dict = {}
        self.contractions_dict = {}
        with open('tables/Contractions.txt', 'r') as contractions, \
                open('tables/EmoticonLookupTable.txt', 'r') as emoticon_table:
            contractions_reader = csv.reader(contractions, delimiter='\t')
            emoticon_reader = csv.reader(emoticon_table, delimiter='\t')
            self.contractions_dict = {rows[0]: rows[1] for rows in contractions_reader}
            self.emoticon_dict = {rows[0]: rows[1] for rows in emoticon_reader}
            contractions.close()
            emoticon_table.close()

        self.grammar = r"""
        NegP: {<VERB>?<ADV>+<VERB|ADJ>?<PRT|ADV><VERB>}
        {<VERB>?<ADV>+<VERB|ADJ>*<ADP|DET>?<ADJ>?<NOUN>?<ADV>?}
        """
        self.chunk_parser = nltk.RegexpParser(self.grammar)
        self.contractions_regex = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))

    def __replace_emoticon(self, text: str) -> str:
        """
        Replace emoticons in the input text with their corresponding words from the emoticon dictionary.

        :param text: The text in which emoticons need to be replaced.
        :return: The text with all emoticons replaced by their corresponding words.
        """
        for k, v in self.emoticon_dict.items():
            text = text.replace(k, v)
        return text

    def tokenize_and_stem(self, text: str) -> list:
        """
        Tokenize the input text into individual words and stem each word.

        :param text: The text to be tokenized and stemmed.
        :return: A list of stemmed tokens.
        """
        return [self.stemmer.stem(t) for t in nltk.word_tokenize(text)]

    def __remove_url(self, text: str) -> str:
        """
        Remove all URLs from the input text.

        :param text: The text from which URLs need to be removed.
        :return: The text with all URLs removed.
        """
        return self.url_regex.sub(' ', text)

    def __negated(self, input_words: List[str]) -> bool:
        """
        Check if the input words contain any negation words.

        :param input_words: The words to be checked.
        :return: True if any of the input words are negation words, False otherwise.
        """
        return len(set(input_words) & set(self.negation_words)) > 0

    def __prepend_not(self, word: str) -> str:
        """
        Prepend 'NOT_' to the input word unless it is a negation word or an emoticon.

        :param word: The word to which 'NOT_' may be prepended.
        :return: The word with 'NOT_' prepended if it is not a negation word or an emoticon, original word otherwise.
        """
        return word if word in self.negation_words or word in self.emoticon_words else 'NOT_' + word

    def __handle_negation(self, text: str) -> str:
        """
        Handle negation in the input text by prepending 'NOT_' to each word that is negated.

        :param text: The text in which negation should be handled.
        :return: The text with 'NOT_' prepended to each word that is negated.
        """
        sentences = nltk.sent_tokenize(text)
        modified_sentences = []
        for st in sentences:
            all_words = nltk.word_tokenize(st)
            modified_words = []
            if self.__negated(all_words):
                part_of_speech = nltk.tag.pos_tag(all_words, tagset='universal')
                chunked = self.chunk_parser.parse(part_of_speech)
                for n in chunked:
                    if isinstance(n, nltk.tree.Tree):
                        words = [pair[0] for pair in n.leaves()]
                        if n.label() == 'NegP' and self.__negated(words):
                            for i, (word, pos) in enumerate(n.leaves()):
                                if (pos == 'ADV' or pos == 'ADJ' or pos == 'VERB') and (word != 'not'):
                                    modified_words.append(self.__prepend_not(word))
                                else:
                                    modified_words.append(word)
                        else:
                            modified_words.extend(words)
                    else:
                        modified_words.append(n[0])
                new_sentence = ' '.join(modified_words)
                modified_sentences.append(new_sentence)
            else:
                modified_sentences.append(st)
        return '. '.join(modified_sentences)

    def __expand_contractions(self, text: str) -> str:
        """
        Expand the contractions in the provided text. For example, "isn't" becomes "is not".

        :param text: The text in which contractions need to be expanded.
        :return: The text with all contractions expanded.
        """

        def replace(match):
            return self.contractions_dict[match.group(0)]

        return self.contractions_regex.sub(replace, text.lower())

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the given text by removing newline characters, ignoring non-ASCII characters,
        expanding contractions, removing URLs, replacing emoticons, and handling negations.

        :param text: The text to be preprocessed.
        :return: The preprocessed text.
        """
        text = text.strip('\n')
        text = text.encode('ascii', 'ignore').decode()
        text = self.__expand_contractions(text)
        text = self.__remove_url(text)
        text = self.__replace_emoticon(text)
        text = self.__handle_negation(text)
        return text


class SentimentAnalyzer:
    def __init__(self, nlp: NLP, use_openai_emb_api: bool):
        self.nlp = nlp
        self.models = {}
        self.use_openai_emb_api = use_openai_emb_api
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
        self.vectorizer = None
        self.optional_apis = ['text-embedding-ada-002',
                              'text-similarity-ada-001',
                              'text-similarity-babbage-001',
                              'text-similarity-curie-001',
                              'text-similarity-davinci-001']

    def train(self, algo: str, x_train: List[str], y_train: List[str], embedder: str = ''):
        """
        Train a machine learning model using the specified algorithm and embedding method on the provided training data.

        :param algo: The name of the machine learning algorithm to use for training.
        :param x_train: The training data, a list of texts to be processed and passed to the model for training.
        :param y_train: The labels corresponding to the training data.
        :param embedder: The name of the method to use for embedding the training data. Defaults to 'TF-IDF'.
        :return: The trained machine learning model.
        """
        if not self.use_openai_emb_api:
            embedder = 'TF-IDF'
        elif embedder not in self.optional_apis:
            print('Invalid openai api name!')
            print('You can choose from the following api names:')
            for api in self.optional_apis:
                print(f'- {api}')
            return
        model = self.algos[algo]
        if not self.use_openai_emb_api:
            x_train = [self.nlp.preprocess_text(text) for text in x_train]
        x_train = self.__get_embeddings(x_train, embedder, mode='train')
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        model.fit(x_train, y_train)
        curr_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if self.use_openai_emb_api:
            with open(f'./model/{algo}-{embedder}-{curr_time}.pkl', 'wb') as model_file:
                pickle.dump(model, model_file)
        else:
            with open(f'./model/{algo}-local-{curr_time}.pkl', 'wb') as model_file:
                pickle.dump(model, model_file)
            with open(f'./model/embedding-{curr_time}.pkl', 'wb') as emb_file:
                pickle.dump(self.vectorizer, emb_file)
        return model

    def __load_model(self, algo, embedder) -> bool:
        """
        Load the machine learning model and corresponding embedding model (if necessary) from the local directory.

        :param algo: The name of the machine learning algorithm that the model was trained with.
        :param embedder: The name of the method used for embedding when training the model.
        :return: True if the model (and embedding model, if necessary) is successfully loaded, False otherwise.
        """
        if algo in self.models.keys() and ((self.use_openai_emb_api and self.vectorizer is None) or (
                not self.use_openai_emb_api and self.vectorizer is not None)):
            return True

        all_files = os.listdir('./model/')
        if self.use_openai_emb_api:
            classifier_models = sorted([f for f in all_files if algo in f and embedder in f], reverse=True)
            classifier_model = classifier_models[0] if len(classifier_models) > 0 else None
            if classifier_model is not None:
                with open(f'./model/{classifier_model}', 'rb') as f:
                    self.models[algo] = pickle.load(f)
                print(f'Model loaded: {classifier_model}')
                return True
            return False
        else:
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
                print(f'Model loaded: {classifier_model}')
                print(f'Model loaded: {embedding_model}')
                return True
            return False

    @staticmethod
    def read_data(file_name, text_index=1, label_index=2) -> Tuple[List[str], List[str]]:
        """
        Read data from a file and extract text and label information based on provided indices.

        :param file_name: The name (and path if necessary) of the file to be read.
        :param text_index: The index of the text in each line of the file. Defaults to 1.
        :param label_index: The index of the label in each line of the file. Defaults to 2.
        :return: A list of texts and a list of corresponding labels.
        """
        with open(file_name, encoding='utf-8', mode='r') as file:
            file_reader = csv.reader(file, delimiter='\t')
            dataset = [(line[text_index], line[label_index]) for line in file_reader]
            texts = [data[0] for data in dataset]
            labels = [data[1] for data in dataset]
            file.close()
        return texts, labels

    @staticmethod
    def gen_dataset(texts_list: List[List[str]], labels_list: List[List[str]]) \
            -> Tuple[List[str], List[str]]:
        """
        Generate a large dataset from small datasets (a list of texts and their corresponding labels).

        :param texts_list: A list of lists of texts. Each inner list represents texts in certain dataset.
        :param labels_list: A list of lists of labels. Each inner list represents the labels corresponding
        to the texts in the same position in texts_list.
        :return: Two lists: the first one is a list of all texts, and the second one is a list of all labels.
        """
        texts = []
        labels = []
        for sub_texts in texts_list:
            texts.extend(sub_texts)
        for sub_labels in labels_list:
            labels.extend(sub_labels)
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

    def __get_embedding(self, text, embedder) -> np.ndarray:
        """
        Generate the embedding for a given text using a specified embedder.

        :param text: The text to be processed and embedded.
        :param embedder: The method used for generating the embedding.
        :return: The generated embedding for the given text.
        """
        if embedder == 'TF-IDF':
            assert self.vectorizer is not None
            return self.vectorizer.transform([text]).toarray()
        return np.array([get_embedding(text=text, engine=embedder)])

    def __get_embeddings(self, texts, embedder, mode='test') -> np.ndarray:
        """
        Generate the embeddings for a list of texts using a specified embedder.

        :param texts: The list of texts to be processed and embedded.
        :param embedder: The method used for generating the embeddings.
        :param mode: The mode of operation. If 'train' is specified, the vectorizer is fitted on the provided texts.
        :return: The generated embeddings for the given list of texts.
        """
        if embedder == 'TF-IDF':
            if mode == 'train':
                self.vectorizer = TfidfVectorizer(tokenizer=self.nlp.tokenize_and_stem, sublinear_tf=True, max_df=0.5,
                                                  stop_words=self.nlp.stop_words, min_df=3)
                return self.vectorizer.fit_transform(texts).toarray()
            else:
                assert self.vectorizer is not None
                return self.vectorizer.transform(texts).toarray()
        else:
            assert self.use_openai_emb_api and embedder in self.optional_apis
            openai_max_token = 2048
            if len(texts) <= openai_max_token:
                return np.array(get_embeddings(list_of_text=texts, engine=embedder))
            embeddings = []
            start = 0
            end = openai_max_token
            while len(texts) >= end != start:
                embeds = get_embeddings(list_of_text=texts[start:end], engine=embedder)
                embeddings.extend(embeds)
                start = end
                end = min(start + openai_max_token, len(texts))
            return np.array(embeddings)

    def run(self, algo: str, text: str, embedder: str = ''):
        """
        Use a specified algorithm to predict the label of a given text.

        :param algo: The name of the machine learning algorithm to use for prediction.
        :param text: The text to be processed and passed to the model for prediction.
        :param embedder: The name of the method to use for embedding the text. Defaults to 'TF-IDF'.
        """
        if not self.use_openai_emb_api:
            embedder = 'TF-IDF'
        if not self.__load_model(algo, embedder):
            print('Fail to load model!')
            return
        else:
            classifier = self.models[algo]
            text = self.nlp.preprocess_text(text)
            feature_vector = self.__get_embedding(text, embedder)
            result = classifier.predict(feature_vector)
            print(f'result: {result[0]}')

    def test(self, algo: str, texts: List[str], y_true: List[str], embedder: str = ''):
        """
        Test a specified algorithm using provided texts and true labels,
        prints out various scores related to the model's performance.

        :param algo: The name of the machine learning algorithm to use for prediction.
        :param texts: A list of texts to be processed and passed to the model for prediction.
        :param y_true: The true labels for the provided texts. This is used for evaluating the model's performance.
        :param embedder: The name of the method to use for embedding the texts. Defaults to 'TF-IDF'.
        :return: precision, recall, F-measure and accuracy.
        """
        if not self.use_openai_emb_api:
            embedder = 'TF-IDF'
        if not self.__load_model(algo, embedder):
            print('Fail to load model!')
            return None, None, None, None
        else:
            classifier = self.models[algo]
            if not self.use_openai_emb_api:
                texts = [self.nlp.preprocess_text(text) for text in texts]
            embeddings = self.__get_embeddings(texts, embedder)
            y_pred = classifier.predict(embeddings)
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            precision = precision_score(y_true, y_pred, average=None)
            recall = recall_score(y_true, y_pred, average=None)
            f1score = f1_score(y_true, y_pred, average=None)
            accuracy = accuracy_score(y_true, y_pred)
            print(f'Precision: {precision}')
            print(f'Recall:    {recall}')
            print(f'F-measure: {f1score}')
            print(f'Accuracy:  {accuracy}')
            print(f'Classification Report:\n{classification_report(y_true, y_pred, digits=4)}')
            return precision, recall, f1score, accuracy

    def reset(self):
        """
        Reset the state of the instance by clearing all loaded models and vectorizer.
        """
        self.models = {}
        self.vectorizer = None
        print('Analyzer reset.')


def main():
    nlp = NLP()
    analyzer = SentimentAnalyzer(nlp, use_openai_emb_api=True)

    texts_1, labels_1 = analyzer.read_data('dataset/se-appreview.txt')
    texts_2, labels_2 = analyzer.read_data('dataset/se-sof4423.txt')
    texts, labels = analyzer.gen_dataset([texts_1, texts_2], [labels_1, labels_2])
    texts_train, texts_test, labels_train, labels_test = analyzer.split_dataset(texts, labels)

    analyzer.train('SGD', texts_train, labels_train, embedder='text-embedding-ada-002')
    analyzer.test('SGD', texts_test, labels_test, embedder='text-embedding-ada-002')

    analyzer.reset()


if __name__ == '__main__':
    main()
