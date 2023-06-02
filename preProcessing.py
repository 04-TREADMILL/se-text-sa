import string
import nltk
from nltk.stem.snowball import SnowballStemmer
import spacy
import csv
import re
from typing import List


class PreProcessor:
    def __init__(self):
        self.model = spacy.load("en_core_web_sm")
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

    def tokenize_and_stem(self, text: str) -> list:
        """
        Tokenize the input text into individual words and stem each word.

        :param text: The text to be tokenized and stemmed.
        :return: A list of stemmed tokens.
        """
        return [self.stemmer.stem(t) for t in nltk.word_tokenize(text)]

    def __replace_emoticon(self, text: str) -> str:
        """
        Replace emoticons in the input text with their corresponding words from the emoticon dictionary.

        :param text: The text in which emoticons need to be replaced.
        :return: The text with all emoticons replaced by their corresponding words.
        """
        for k, v in self.emoticon_dict.items():
            text = text.replace(k, v)
        return text

    def __expand_contractions(self, text: str) -> str:
        """
        Expand the contractions in the provided text. For example, "isn't" becomes "is not".

        :param text: The text in which contractions need to be expanded.
        :return: The text with all contractions expanded.
        """

        def replace(match):
            return self.contractions_dict[match.group(0)]

        return self.contractions_regex.sub(replace, text.lower())

    def __remove_url(self, text: str) -> str:
        """
        Remove all URLs from the input text.

        :param text: The text from which URLs need to be removed.
        :return: The text with all URLs removed.
        """
        return self.url_regex.sub(' ', text)

    def __remove_md_pic(self, text: str) -> str:

        return re.sub(r"![.*?](.*?)", " ", text)

    def __remove_html_tag(self, text: str) -> str:

        return re.sub(r"<(\w+)>.*</\1>", " ", text)

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

    def replace_PROPN(self, text: str) -> str:
        doc = self.model(text)
        new_txt = ""
        for token in doc:
            if token.pos_ == "PROPN":
                new_txt += " PROPN"
            else:
                new_txt += token.text if token.text in string.punctuation else " " + token.text
        return new_txt

    def replace_entity(self, text: str) -> str:
        doc = self.model(text)
        new_txt = text
        for ent in doc.ents:
            new_txt.replace(ent.text, ent.label_)
        return new_txt

    def add_prefix_to_domain_specific_words(self, text: str, prefix: str = "neutral_") -> str:
        """
        给所有软工领域特定词汇加上"neutral_"的前缀
        :param text: 需要进行处理的段落
        :param prefix: 要加上的前缀，默认为"neutral_"
        :return: 加上前缀之后的字符串
        """
        # "value", "values", "default", "dead",
        domain_specific_words = ["support", "supported", "supporting",
                                 "block", "error", "bug"]
        word_list = text.split()
        for i in range(len(word_list)):
            if word_list[i].lower() in domain_specific_words:
                word_list[i] = prefix + word_list[i]
        modified_string = ' '.join(word_list)
        return modified_string

    def add_prefix_to_italic_and_uppercase(self, text: str, prefix: str = "strong_"):
        """
        给大写和斜体的单词加上前缀"strong_"
        :param text: 需要进行处理的段落
        :param prefix: 要加上的前缀，默认为"strong_"
        :return: 加上前缀之后的字符串
        """
        # 匹配斜体单词的正则表达式
        italic_pattern = r'\*([^*]+)\*'
        # 匹配大写单词的正则表达式
        uppercase_pattern = r'\b([A-Z]+)\b'

        # 给斜体单词加上前缀
        text = re.sub(italic_pattern, r'{0}\1'.format(prefix), text)
        # 给大写单词加上前缀
        text = re.sub(uppercase_pattern, r'{0}\1'.format(prefix), text)

        return text

    def preprocess_text(self, text: str) -> str:
        text = text.strip('\n')
        text = text.encode('ascii', 'ignore').decode()
        text = self.__expand_contractions(text)
        text = self.__remove_url(text)
        text = self.__replace_emoticon(text)
        text = self.__handle_negation(text)
        text = self.__remove_md_pic(text)
        text = self.__remove_html_tag(text)
        # sjy
        # text = self.replace_PROPN(text)
        # text = self.replace_entity(text)
        # mys
        # text = self.add_prefix_to_domain_specific_words(text)
        # text = self.add_prefix_to_italic_and_uppercase(text)
        return text

    def preprocess_text_v2(self, text: str) -> str:
        text = text.strip('\n').strip(' ')
        text = self.__remove_url(text)
        text = self.__remove_md_pic(text)
        text = self.__remove_html_tag(text)
        return text


if __name__ == "__main__":
    pp = PreProcessor()
    text1 = "This is a fatal error."
    text2 = "This version is no longer supported."
    text3 = "This is *very* IMPORTANT"
    new1 = pp.add_prefix_to_domain_specific_words(text1)
    new2 = pp.add_prefix_to_domain_specific_words(text2)
    new3 = pp.add_prefix_to_italic_and_uppercase(text3)
    print(new1, new2, new3)

    # txt = "In which case, shouldn't the code just say that all other open projects should already have their SDK's pointing to the SDK used by the IDE?  i.e., now that we do this, we shouldn't have to modify other projects, only this project since all others should already be in sync.  The fishy part here is just that choosing something in a project import dialog affects other projects which doesn't make sense. If you want that experience, then the dialog should be more of a global IDE preference."
    # new1 = pp.replace_PROPN(txt)
    # print(new1)
    # new2 = pp.replace_entity(txt)
    # print(new2)
    # new_ = pp.replace_entity(new1)
    # print(new_)
