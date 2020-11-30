import argparse

from ModelClasses.fasttext_model import FasttextModel
from ModelClasses.bert_model import BertModel
from ModelClasses.word2vec_model import Word2vecModel
from Helpers.parameters import *

training_algo = [FASTTEXT, W2V, BERT]


def main(args):
    """This main function should run in order to activate the CLI"""
    # training
    lang = args.lang
    corpus = args.corpus
    if args.algo in [W2V, BERT]:
        # lang = False if int(args.lang) == 1 else True
        if args.algo == W2V:
            if corpus is None:
                Word2vecModel(hebrew_model=(lang == HEBREW)).train()
            else:
                Word2vecModel(hebrew_model=(lang == HEBREW)).train(corpus)
        if args.algo == BERT:
            if corpus is None:
                BertModel(hebrew_model=(lang == HEBREW)).train()
            else:
                BertModel(hebrew_model=(lang == HEBREW)).train(corpus)

    elif args.algo == FASTTEXT:
        if corpus is None:
            FasttextModel().train()
        else:
            FasttextModel().train(corpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line Interface for training term set Synonym Search')
    parser.add_argument('--algo', choices=training_algo,
                        help=f'Choose the algorithm you want to train: insert algo name: \n' + str(training_algo))
    parser.add_argument('--corpus', type=str, default=None,
                        help='Enter the path of the corpus : skip for the default one')
    parser.add_argument('--lang', type=str, choices=[ENGLISH, HEBREW], help='insert the language: english , hebrew')
    args = parser.parse_args()
    main(args)
