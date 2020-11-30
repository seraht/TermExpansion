"""
Authors: ST & MM
Fasttext class implementation
"""

from TrainingModels.TrainingCode.Training_Word2Vec import *
from ModelClasses.model import Model
import Helpers.config as conf
import ModelEvaluation.mAP as Evaluation
import WordExpansion.Synonyms as Syn


class Word2vecModel(Model):
    def __init__(self, path=None, hebrew_model=False):
        """
        :param name: Model Name
        :param hebrew_model: Language of the model
        """
        self.hebrew_model = hebrew_model
        super().__init__(conf.MODEL_TYPE['word2vec'], path)

    def train(self, corpus_path=conf.REPORTS_SAMPLE_PATH):
        """
        Training the word2vec model
        :param corpus_path: Corpus to train on
        :return: model name & entry point
        """
        if self.path is None:
            self.path = train_word2vec(corpus_path, hebrew_model=self.hebrew_model)
        else:
            raise ValueError('Only new word2vec models can be trained')

    def expand_terms(self, finding, corpus=conf.REPORTS_SAMPLE_PATH, top_n=10):
        """
        Get synonyms and frequencies for a given Finding name
        :param finding: Finding name, term to look for
        :param corpus: Corpus to check to check the frequency
        :param top_n: Number of synonym
        :return:
        """
        result = Syn.get_word2vec(finding, self.path, corpus, top_n=top_n)
        return result

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
