"""
Authors: ST & MM
Fasttext class implementation
"""

from TrainingModels.TrainingCode.Training_Bert import train_bert
from ModelClasses.model import Model
import Helpers.config as conf
import ModelEvaluation.mAP as Evaluation
import WordExpansion.Synonyms as Syn


class BertModel(Model):
    def __init__(self, path=None, hebrew_model=False):
        """
        :param name: Name of the model
        :param hebrew_model: language of the model
        """
        self.hebrew_model = hebrew_model
        super().__init__(conf.MODEL_TYPE['bert'], path)

    def train(self, corpus_path=conf.REPORTS_SAMPLE_PATH):
        """
        Train function for the model
        :param corpus_path: Corpus path to train the model on
        :return: corpus name & entry point
        """
        if self.path is None:
            self.path = train_bert(corpus_path, self.hebrew_model)

    def expand_terms(self, finding, corpus_path=conf.REPORTS_SAMPLE_PATH, top_n=10):
        """
        Getting synonyms of a finding name
        :param finding: finding name, term too look for
        :param corpus_path: Corpus path to train the model on
        :param top_n: number of synonyms
        :return: synonyms
        """
        synonyms = Syn.get_bert(finding, self.path, corpus_path, top_n=top_n, hebrew_model=self.hebrew_model)
        return synonyms

    def evaluate_term_map(self, finding, corpus=conf.REPORTS_SAMPLE_PATH):
        """
        Evaluation for a specific finding name
        You can chose only findings that are in the gold standard list
        :param finding: finding name, term to look for
        :param corpus: corpus to generate sentences with the given term
        :return: Evaluation score
        """
        return Evaluation.evaluate_term_map(finding, self.type, self.path, corpus=corpus, hebrew_model=self.hebrew_model)

    def evaluate_algorithm(self, corpus=conf.REPORTS_SAMPLE_PATH):
        """
        Evaluation of the model
        :param corpus: corpus to generate sentences with the given term
        :return: Evaluation score
        """
        return Evaluation.evaluate_algorithm(self.type, model_path=self.path, corpus=corpus)
