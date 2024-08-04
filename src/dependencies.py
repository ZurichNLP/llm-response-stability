import numpy as np
import spacy.attrs
import re

class DependencyParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.nlp.tokenizer = spacy.tokenizer.Tokenizer(
            self.nlp.vocab,
            token_match=re.compile(r"\S+").match,
        )
        self.tagger = self.nlp.get_pipe("tagger")

    def parse_dependency(self, text):
        doc = self.nlp(text)

        n_rights = np.mean([token.n_rights for token in doc])
        n_lefts = np.mean([token.n_lefts for token in doc])
        dep_distance = np.mean([abs(token.i - token.head.i) for token in doc])
        return n_rights, n_lefts, dep_distance
