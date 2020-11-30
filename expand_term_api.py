import argparse
import Helpers.helpers as help
import Helpers.config as conf
from ModelClasses.fasttext_model import FasttextModel
from ModelClasses.bert_model import BertModel
from ModelClasses.word2vec_model import Word2vecModel
import WordExpansion.Synonyms as Syn
from Helpers.parameters import *

algos = [W2V, FASTTEXT, BERT, UMLS, WORDNET, BABELNET]

def main(args):
    model = args.algo
    corpus = args.corpus
    lang = args.lang
    finding = args.finding
    algo_path = args.algo_path
    if corpus is None:
        corpus = conf.REPORTS_SAMPLE_PATH
    if model == BERT:
        synonyms = BertModel(algo_path, hebrew_model=(lang == HEBREW)).expand_terms(finding, corpus)
    elif model == BABELNET:
        synonyms = Syn.expand_term_api(BABELNET, finding, corpus, hebrew_api=(lang == HEBREW))
    elif model == FASTTEXT:
        synonyms = FasttextModel(algo_path).expand_terms(finding, corpus)
    elif model == W2V:
        synonyms = Word2vecModel(algo_path, hebrew_model=(lang == HEBREW)).expand_terms(finding, corpus)
    else:
        synonyms = Syn.expand_term_api(model, finding, corpus)
    help.display_beautiful_list(synonyms)

    return synonyms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line Interface for expanding term')
    parser.add_argument('--algo', type=str, choices=algos, required=True,
                        help=f'Choose the algorithm you want to expand term from: insert algo name: \n' + str(algos))
    parser.add_argument('--algo_path', type=str,
                        help='For pretrained algorithms choose a path among the available algorithms in /Models')
    parser.add_argument('--finding', type=str, required=True, help="Finding name to generate synonyms from")
    parser.add_argument('--corpus', type=str, default=None,
                        help='Enter the path of the corpus : skip for the default one')
    parser.add_argument('--lang', type=str, choices=[ENGLISH, HEBREW],  help='insert the language: english , hebrew')
    args = parser.parse_args()
    main(args)


