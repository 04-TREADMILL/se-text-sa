import string
import re
import spacy


class PreProcessor:
    def __init__(self):
        self.model = spacy.load("en_core_web_sm")
        print("Preprocessor initialized!")

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
        domain_specific_words = ["value", "values", "support", "supported", "supporting", "dead",
                                 "block", "default", "error", "bug"]
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
