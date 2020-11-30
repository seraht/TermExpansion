"""
Authors: ST & MM
Bert train
"""

import Helpers.config as conf
from Helpers.helpers import get_corpus_name, get_date_for_model_name

import os
import shutil
import re
import transformers
import pandas as pd
from collections import Counter
import torch
from gensim.models.word2vec import Text8Corpus
from gensim.models.phrases import Phrases, Phraser


def train_bert(corpus_path, hebrew_model=False):
    """
    Bert model training
    :param corpus_path: Corpus to train Bert on
    :param hebrew_model: Model in Hebrew or not
    :return: The name of the new trained model
    """
    language = 'hebrew' if hebrew_model else 'english'
    df = pd.read_csv(corpus_path)
    corpus_name = get_corpus_name(corpus_path)
    print("Preprocess...")
    if hebrew_model:
        model_name, vocab, raw_text_file = preprocess_hebrew(df, corpus_name)
    else:
        model_name, vocab, raw_text_file = preprocess_english(df, corpus_name)
        pass

    print("Cuda availability :", torch.cuda.is_available())
    print("Getting tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(conf.bert_model[language], use_fast=True)
    model = transformers.AutoModelForMaskedLM.from_pretrained(conf.bert_model[language]).to('cuda')

    tokenizer.add_tokens(vocab)
    model.resize_token_embeddings(len(tokenizer))

    if os.path.exists(conf.MODELS_PATH + conf.TYPE_MODEL_PATH['bert'] + model_name):
        shutil.rmtree(conf.MODELS_PATH + conf.TYPE_MODEL_PATH['bert'] + model_name)

    os.mkdir(conf.MODELS_PATH + conf.TYPE_MODEL_PATH['bert'] + model_name)
    tokenizer.save_pretrained(conf.MODELS_PATH + conf.TYPE_MODEL_PATH['bert'] + model_name)

    print("Tokenizing...")
    dataset = transformers.LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=raw_text_file,
        block_size=128,
    )

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = transformers.TrainingArguments(
        output_dir=conf.MODELS_PATH + conf.TYPE_MODEL_PATH['bert'] + model_name,
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        save_steps=300,
        logging_steps=100,
        save_total_limit=3,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    print("Begin training...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer.train()
    trainer.save_model(conf.MODELS_PATH + conf.TYPE_MODEL_PATH['bert'] + model_name)
    print('The model has been saved under : ', conf.MODELS_PATH + conf.TYPE_MODEL_PATH['bert'] + model_name)

    return conf.MODELS_PATH + conf.TYPE_MODEL_PATH['bert'] + model_name

def preprocess_hebrew(reports_df, corpus_name):
    """
    Hebrew reports preprocessing
    :param reports_df: dataframe of hebrew reports
    :param corpus_name: Name of the corpus
    :return: model name, vocab list, model_name, raw_text_file to tokenize on
    """
    reports_df = reports_df[conf.col_report][reports_df[conf.col_source] == conf.HEBREW_REPORT_INDICATOR]

    # preprocess
    reports_df = reports_df.apply(sub_preporcess_hebrew)
    text_reports = reports_df.values
    text_reports = ' '.join(list(text_reports)).encode('utf-8')

    # Save raw text
    date = get_date_for_model_name()
    model_name = date + '_bert_' + conf.HEBREW_IDENTIFIER_NAME + '_' + corpus_name
    raw_text_file = conf.DATA_PATH + model_name + '.txt'

    if os.path.exists(raw_text_file):
        os.remove(raw_text_file)

    with open(raw_text_file, "wb") as text_file:
        text_file.write(text_reports)

    # Load training data.
    sentences = Text8Corpus(raw_text_file)

    # Train a toy bigram model.
    phrases = Phrases(sentences, min_count=7, threshold=100, max_vocab_size=len(text_reports))
    del text_reports

    # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    bigram = Phraser(phrases)

    reports_df = reports_df.str.split(' ')
    reports_df = reports_df.apply(lambda x: bigram[x])
    reports_df = reports_df.apply(lambda x: " ".join(x))

    reports_df.to_csv(raw_text_file, header=None, index=None, sep=' ', mode='a')
    del reports_df

    counter = Counter()
    with open(raw_text_file, encoding="utf-8") as f:
        for line in f:
            counter.update(line.split())

    vocab = list(counter.keys())
    return model_name, vocab, raw_text_file

def sub_preporcess_hebrew(text):
    """
    Hebrew reports preprocess
    :param text: report
    :return: preprocessed report
    """
    text = re.sub(r"[a-zA-Z]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", " ", text)
    text = re.sub(r"(\u200e)", " ", text)
    text = re.sub(r"(\u200f)", " ", text)
    text = re.sub(r"(\n)", " ", text)
    text = re.sub(r"(\r)", "", text)
    return text

def preprocess_english(reports_df, corpus_name):
    """
    English reports preprocessing
    :param reports_df: dataframe of hebrew reports
    :param corpus_name: Name of the corpus
    :return: model name, vocab list, model_name, raw_text_file to tokenize on
    """
    # Getting and cleaning bigrams
    ontology_df = pd.read_csv(conf.ontology_file)
    ontologies = list(ontology_df[conf.finding_col])
    bigrams = [ontology for ontology in ontologies if '_' in ontology]
    clean_bigrams = [re.sub(r'_', ' ', bigram) for bigram in bigrams]

    # preprocess
    reports_df = reports_df.loc[reports_df[conf.col_source] != conf.HEBREW_REPORT_INDICATOR]
    reports = reports_df[[conf.col_report]]
    reports[conf.col_report] = reports[conf.col_report].str.lower()
    reports[conf.col_report] = reports[conf.col_report].apply(sub_preprocess_english)

    def report_to_bigrams(report):
        """
        Sub-function to transform the bigrams into processable processable bigrams
        :param report: Medical report
        :return: Report with processable bigrams
        """
        for i, bigram in enumerate(clean_bigrams):
            report = re.sub(bigram, bigrams[i], report)
        return report

    reports[conf.col_report] = reports[conf.col_report].apply(report_to_bigrams)

    # Save raw text
    date = get_date_for_model_name()
    model_name = date + '_bert_' + conf.ENGLISH_IDENTIFIER_NAME + '_' + corpus_name
    raw_text_file = conf.DATA_PATH + model_name + '.txt'

    reports.to_csv(raw_text_file, header=None, index=None, sep=' ', mode='a')

    counter = Counter()
    with open(raw_text_file) as f:
        for line in f:
            counter.update(line.split())

    vocab = []
    for keys, values in counter.items():
        if 100 < values < 10000:
            vocab.append(keys)

    vocab = [re.sub(r'\.', '', word) for word in vocab]

    return model_name, vocab, raw_text_file


def sub_preprocess_english(text):
    """
    English reports preprocess
    :param text: report
    :return: preprocessed report
    """
    processed_text = re.sub(r"\n", ' ', text)
    processed_text = re.sub(r"[^a-zA-Z\.]+", ' ', processed_text)
    processed_text = re.sub(r"\s\.", '.', processed_text)
    return processed_text
