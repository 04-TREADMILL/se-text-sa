import string

import spacy


class PreProcessor:
    def __init__(self):
        self.model = spacy.load("en_core_web_sm")

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


if __name__ == "__main__":
    pp = PreProcessor()
    txt = "In which case, shouldn't the code just say that all other open projects should already have their SDK's pointing to the SDK used by the IDE?  i.e., now that we do this, we shouldn't have to modify other projects, only this project since all others should already be in sync.  The fishy part here is just that choosing something in a project import dialog affects other projects which doesn't make sense. If you want that experience, then the dialog should be more of a global IDE preference."
    new1 = pp.replace_PROPN(txt)
    print(new1)
    new2 = pp.replace_entity(txt)
    print(new2)
    new_ = pp.replace_entity(new1)
    print(new_)
