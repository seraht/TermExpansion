"""
Authors: ST & MM
Fasttext class implementation
"""

from TrainingModels.TrainingCode.Training_Fasttext import *
from ModelClasses.model import Model
import Helpers.config as conf
import ModelEvaluation.mAP as Evaluation
import WordExpansion.Synonyms as Syn


class FasttextModel(Model):
    def __init__(self, path=None):
        """
        :param name: Model Name
        """
        super().__init__(conf.MODEL_TYPE['fasttext'], path)

    def train(self, corpus_path=conf.REPORTS_SAMPLE_PATH):
        """
        Train function for the model
        :param corpus_path: Corpus path to train the model on
        :return:
        """
        if self.path is None:
            self.path = train_fasttext(corpus_path)
        else:
            raise ValueError('Only new fasttext models can be trained')

    def expand_terms(self, finding, corpus=conf.REPORTS_SAMPLE_PATH, top_n=10):
        """
        Get synonyms and frequencies for a given Finding name
        :param finding: Finding name, term to look for
        :param corpus: Corpus to check to check the frequency
        :param top_n: Number of synonym
        :return:
        """
        synonyms = Syn.get_fasttext(finding, self.path, corpus, top_n=top_n)
        return synonyms

    def evaluate_term_map(self, finding):
        """
        Get evaluation for a a given finding term
        You can chose only findings that are in the gold standard list
        :param finding: Finding name, term to look for
        :return:Evaluation score for the given term
        """
        return Evaluation.evaluate_term_map(finding, self.type, self.path)

    def evaluate_algorithm(self):
        """
        Evaluates an algorithm on a given gold standard list
        :return: Evaluation scores
        """
        return Evaluation.evaluate_algorithm(self.type, model_path=self.path)
