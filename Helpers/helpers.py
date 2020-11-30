"""
Authors Serah Tapia & Michael Marcus
Helpers pieces of code for the term_expension api
"""
import re

from datetime import datetime

# General helpers
def get_corpus_name(corpus_path):
    """
    Getting the the name of a corpus from a path
    :param corpus_path:
    :return: corpus name
    """
    return re.search(r'^\/(.+\/)*(.+)\.(.+)$', '/' + corpus_path).group(2)

def get_date_for_model_name():
    """
    Generates today's date and incorporate in order to incorporate it in the model's name
    :return: today's date with the following format : 'ddmmyy
    """
    now = datetime.now()
    year = now.strftime("%y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    return day + month + year

def display_beautiful_list(array):
    """
    Display beautifully a list
    :param array: list in entry
    :return: None
    """
    for i, content in enumerate(array):
        print(str(i + 1) + '. ', content)

# Preprocessing helpers
def preprocess_he(text):
    """
    preprocess for reports in Hebrew
    :param text: original report
    :return: preprocessed report
    """
    text = re.sub(r"[a-zA-Z]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", " ", text)
    text = re.sub(r"(\u200e)", " ", text)
    text = re.sub(r"(\u200f)", " ", text)
    text = re.sub(r"(\n)", " ", text)
    text = re.sub(r"(\r)", "", text)
    text = re.sub(r' +', ' ', text, flags=re.I)
    return text


def preprocess_en(text):
    """
    preprocess for reports in English
    :param text: original report
    :return: preprocessed report
    """
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", " ", text)
    text = re.sub(r"(\n)", " ", text)
    text = re.sub(r"(\r)", "", text)
    return text

def rep_to_bigrams(report, bigrams, clean_bigrams):
    """
    cleaning a report by replacing bigrams with processable bigrams
    :param report: medical report
    :param bigrams: list of bigrams with this format "pleural_effusion"
    :param clean_bigrams: list of bigrams with this format 'pleural effusion"
    :return:
    """
    for i, bigram in enumerate(clean_bigrams):
        report = re.sub(bigram, bigrams[i], report)
    return report
