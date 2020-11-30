"""
Authors: ST & MM
Retrieving positive sentences for testing Bert
"""

import re
import pandas as pd
import numpy as np
from collections import defaultdict
import Helpers.config as conf

def preprocess(text):
    """Preprocessing the report:
    removing numbers, punctuation, extra spaces.."""
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", " ", text)
    text = re.sub(r"(\u200e)", " ", text)
    text = re.sub(r"(\u200f)", " ", text)
    text = re.sub(r"(\n)", " ", text)
    text = re.sub(r"(\r)", "", text)
    text = re.sub(r"( +)", " ", text)
    text = text.lower()
    return text

def preprocess_he(text):
    """Preprocess hebrew reports
    :param text: reports in hebrew
    :return: reports in hebrew without english words, numbers, punctuation, extra spaces...
    """
    text = re.sub(r"[a-zA-Z]", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[!\"#$%&'()*+,-/:;<=>?@[\]^_`{|}~]", " ", text)
    text = re.sub(r"(\u200e)", " ", text)
    text = re.sub(r"(\u200f)", " ", text)
    text = re.sub(r"(\n)", " ", text)
    text = re.sub(r"(\r)", "", text)
    text = re.sub(r' +', ' ', text, flags=re.I)
    return text

def get_findings_en(file=conf.ontology_file, finding_column_name=conf.finding_col):
    """
    Get the list of all the findings
    :param file: csv file with all the findings
    :param finding_column_name: name of the column finding
    :return: List with all the findings
    """
    findings_cxr = pd.read_csv(file)
    findings_cxr.drop_duplicates(inplace=True)
    findings_cxr[finding_column_name] = findings_cxr.finding_name.str.replace('_', ' ')
    findings = findings_cxr[finding_column_name].values
    return findings

def get_findings_he(file=conf.finding_col, finding_column_he=conf.finding_col_he):
    findings_cxr = pd.read_csv(file)
    findings_cxr.drop_duplicates(inplace=True)
    result = findings_cxr[finding_column_he].tolist()
    result = [word for word in result if not isinstance(word, (int, float))]
    words = ','.join(result)
    output = []
    words = words.split(',')
    for word in words:
        word = re.sub(r"[!\"#$%&'*+/<=>?@[\]^{|}~]", "", word)
        word = re.sub(r' +', ' ', word, flags=re.I)
        word = word.strip()
        word = word.lower()
        word = word.replace(' ', '_')
        output.append(word)
    return output

def check(sentences, negation):
    """
    Remove all the negative sentences from the list sentences
    :param sentences: list of sentences from the report
    :param negation: list of negative terms
    :return: list of positive sentences from the report
    """
    res = [all([k not in s for k in negation]) for s in sentences]
    return [sentences[i] for i in range(0, len(res)) if res[i]]


def get_sentences_for_finding(sentences, finding):
    """
    Retrieve all the positive sentences which contain a specific finding
    :param sentences: list of sentences from the report
    :param finding: finding to look for
    :return: dict of list
    """
    negation = conf.negation
    sentences = check(sentences, negation)
    sentences_finding = []
    for sent in sentences:
        if finding in sent and len(sent) > 10:
            sentences_finding.append(sent)
    return sentences_finding

def select_sentences_for_finding(corpus, finding, hebrew_model=False, report_column_name=conf.col_report):
    """
    Select for each finding all the positive sentences that contain it
    :param finding: Finding name, term to look for
    :param corpus: csv file with reports
    :param report_column_name: name of the column report
    :param hebrew_model: Model language
    :return: Dict with the finding in key, and a list of \
    all the positive sentences that contain the finding in the value.
    """
    data = pd.read_csv(corpus)
    reports = '. '.join(list(data[report_column_name].values))
    reports = np.array(reports.split('.'))
    if hebrew_model:
        sentences = np.array(list(map(preprocess_he, reports)))
        result = get_sentences_for_finding(sentences, finding)
    else:
        sentences = np.array(list(map(preprocess, reports)))
        result = get_sentences_for_finding(sentences, finding)
    return result


def get_sentences(sentences, findings):
    """
    Retrieve all the positive sentences which contain a specific finding
    :param sentences: list of sentences from the report
    :param findings: finding to look for
    :return: dict of list
    """
    negation = conf.negation
    sentences = check(sentences, negation)
    sentences_per_finding = defaultdict(list)
    for i in range(len(findings)):
        for sent in sentences:
            if findings[i] in sent and len(sent) > 10:
                sentences_per_finding[findings[i]].append(sent)
    return sentences_per_finding

def select_sentences(corpus, report_column_name=conf.col_report, lang='EN'):
    """
    Select for each finding all the positive sentences that contain it
    :param corpus: csv file with reports
    :param report_column_name: name of the column report
    :param lang: Model language
    :return: Dict with the finding in key, and a list of \
    all the positive sentences that contain the finding in the value.
    """
    data = pd.read_csv(corpus)
    reports = '. '.join(list(data[report_column_name].values))
    reports = np.array(reports.split('.'))
    if lang == 'EN':
        sentences = np.array(list(map(preprocess, reports)))
        result = get_sentences(sentences, get_findings_en())
    else:
        sentences = np.array(list(map(preprocess_he, reports)))
        result = get_sentences(sentences, get_findings_he())
    return result

def output_csv(corpus, lang):
    """
    Save the sentences of each finding in a csv table
    :param corpus: dict obtained in the output of the function select_sentences()
    :param lang: Model language
    :return: csv file sentences_for_bert.csv
    """
    result_dic = select_sentences(corpus, report_column_name=conf.col_report, lang=lang)
    df = pd.DataFrame.from_dict(result_dic, orient='index')
    df = df.transpose()
    csv_path = conf.ARCHIVES_PATH + 'bert_' + lang + '_pos_sentences.csv'
    df.to_csv(csv_path)
    return
