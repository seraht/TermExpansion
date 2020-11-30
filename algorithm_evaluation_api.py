import argparse
import Helpers.helpers as help
import Helpers.config as conf
from ModelClasses.fasttext_model import FasttextModel
from ModelClasses.bert_model import BertModel
from ModelClasses.word2vec_model import Word2vecModel
import ModelEvaluation.mAP as Map
from Helpers.parameters import *

algos = [W2V, FASTTEXT, BERT, UMLS, WORDNET, BABELNET]


def main(args):
    model = args.algo
    corpus = args.corpus
    lang = args.lang
    finding = args.finding
    algo_path = args.algo_path
    eval_all_terms = args.eval_all_terms
    if corpus is None:
        corpus = conf.REPORTS_SAMPLE_PATH
    if eval_all_terms:
        # Evaluation of all the terms in the default Gold Standard List for a given algorithm
        if model == BERT:
            score = BertModel(algo_path, hebrew_model=(lang == HEBREW)).evaluate_algorithm(corpus)
        elif model == BABELNET:
            score = Map.evaluate_algorithm(BABELNET, hebrew_model=lang)
        elif model == FASTTEXT:
            score = FasttextModel(algo_path).evaluate_algorithm()
        elif model == W2V:
            score = Word2vecModel(algo_path, hebrew_model=(lang == HEBREW)).evaluate_algorithm()
        else:
            score = Map.evaluate_algorithm(model)
        print('mAP Score :', score[0])
        print('median mAP Score :', score[1])

    else:
        # Evaluation of a single given specific term for a given algorithm
        if model == BERT:
            score = BertModel(algo_path, hebrew_model=(lang == HEBREW)).evaluate_term_map(finding, corpus)
        elif model == BABELNET:
            score = Map.evaluate_term_map(finding, BABELNET, hebrew_model=lang)
        elif model == FASTTEXT:
            score = FasttextModel(algo_path).evaluate_term_map(finding)
        elif model == W2V:
            score = Word2vecModel(algo_path, hebrew_model=(lang == HEBREW)).evaluate_term_map(finding)
        else:
            score = Map.evaluate_term_map(finding, model)
        print('mAP Score :', score)

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line Interface for evaluating algorithm')
    parser.add_argument('--algo', type=str, choices=algos, required=True,
                        help=f'Choose the algorithm you want to evaluate: insert algo name: \n' + str(algos))
    parser.add_argument('--algo_path', type=str,
                        help='For pretrained algorithms choose a path among the available algorithms in /Models')
    parser.add_argument('--finding', type=str, help="Used only for single term evaluation")
    parser.add_argument('--corpus', type=str, default=None,
                        help='Enter the path of the corpus : skip for the default one')
    parser.add_argument('--lang', type=str, choices=[ENGLISH, HEBREW],  help='insert the language: english , hebrew')
    parser.add_argument('-A', '--eval_all_terms', action='store_true',
                        help="Option to evaluate the algorithm on the all the default terms")
    args = parser.parse_args()
    main(args)
