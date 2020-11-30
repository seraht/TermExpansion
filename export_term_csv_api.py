import argparse
import Helpers.config as conf
import ExportExcel.export_to_excel as exp
from ModelClasses.fasttext_model import FasttextModel
from ModelClasses.bert_model import BertModel
from ModelClasses.word2vec_model import Word2vecModel

def main(args):
    corpus = args.corpus
    finding = args.finding
    w2v_path = args.word2vec
    bert_path = args.bert
    ft_path = args.fasttext
    if corpus is None:
        corpus = conf.REPORTS_SAMPLE_PATH
    w2v_model = Word2vecModel(w2v_path)
    bert_model = BertModel(bert_path)
    fasttext_model = FasttextModel(ft_path)

    return exp.output_expanded_term_to_excel(finding, bert_model=bert_model, w2v_model=w2v_model,
                                             fstxt_model=fasttext_model, corpus=corpus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command Line Interface for exporting term in CSV')
    parser.add_argument('--word2vec', type=str, required=True, help='Path of pretrained word2vec algorithm')
    parser.add_argument('--bert', type=str, required=True, help='Path of pretrained bert algorithm')
    parser.add_argument('--fasttext', type=str, required=True, help='Path of pretrained fasttext algorithm')
    parser.add_argument('--finding', type=str, required=True, help="Finding name to generate synonyms from")
    parser.add_argument('--corpus', type=str, default=None,
                        help='Enter the path of the corpus : skip for the default one')
    args = parser.parse_args()
    main(args)
