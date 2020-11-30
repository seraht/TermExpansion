"""
Authors: ST & MM
Fasttext train
"""

from Helpers.helpers import get_date_for_model_name, get_corpus_name
import Helpers.config as conf
import re
import os

import pandas as pd
from tqdm import tqdm

import fasttext

fasttext.FastText.eprint = lambda x: None


pd.options.mode.chained_assignment = None
REFERENCE_FILE_PATH = conf.ontology_file

COLS_TO_KEEP = [conf.col_source, conf.col_report]

def rep_to_bigrams(report, bigrams, clean_bigrams):
    """
    Cleaning a report by replacing bigrams with processable bigrams
    :param report: medical report
    :param bigrams: list of bigrams with this format "pleural_effusion"
    :param clean_bigrams: list of bigrams with this format 'pleural effusion"
    :return:
    """
    for i, bigram in enumerate(clean_bigrams):
        report = re.sub(bigram, bigrams[i], report)
    return report

def train_fasttext(corpus_path):
    """
    Fasttext model training
    :param corpus_path: corpus to train on
    :return: Model Name and model entry point
    """
    corpus_name = get_corpus_name(corpus_path)
    ontology_df = pd.read_csv(REFERENCE_FILE_PATH)
    ontologies = list(ontology_df[conf.finding_col])

    bigrams = [ontology for ontology in ontologies if '_' in ontology]
    clean_bigrams = [re.sub(r'_', ' ', bigram) for bigram in bigrams]

    df = pd.read_csv(corpus_path)
    df = df[COLS_TO_KEEP]

    # Keeping only english reports
    df_english = df[df[conf.col_source] != conf.HEBREW_REPORT_INDICATOR]

    # removing useless terms
    df_english[conf.col_report] = df_english['report'].str.lower()
    df_english[conf.col_report] = [re.sub(r'[\n]', ' ', str(x)) for x in df_english[conf.col_report]]
    df_english[conf.col_report] = [re.sub(r'[^A-Za-z]+', ' ', str(x)) for x in df_english[conf.col_report]]
    df_english[conf.col_report] = [re.sub(r'\s\s+', ' ', str(x)) for x in df_english[conf.col_report]]

    # bigrams implementation
    tqdm.pandas()
    df_english[conf.col_report] = df_english[conf.col_report].progress_apply(lambda x: rep_to_bigrams(x, bigrams, clean_bigrams))

    date = get_date_for_model_name()
    model_name = date + '_fasttext_' + corpus_name

    if os.path.exists(conf.DATA_PATH + model_name + '.txt'):
        os.remove(conf.DATA_PATH + model_name + '.txt')

    df_english.report.to_csv(conf.DATA_PATH + model_name + '.txt', header=None, index=None, sep=' ', mode='a')

    # Training
    model = fasttext.train_unsupervised(conf.DATA_PATH + model_name + '.txt', 'skipgram')
    model.save_model(conf.MODELS_PATH + conf.TYPE_MODEL_PATH['fasttext'] + model_name + '.bin')
    print("Model trained and saved in", conf.MODELS_PATH + conf.TYPE_MODEL_PATH['fasttext'] + model_name + '.bin')

    return conf.MODELS_PATH + conf.TYPE_MODEL_PATH['fasttext'] + model_name + '.bin'
