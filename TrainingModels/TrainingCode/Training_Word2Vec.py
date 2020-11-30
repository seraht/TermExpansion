"""
Authors: ST & MM
Word2Vec Train
"""

from gensim.models import Phrases
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
from Helpers.helpers import get_date_for_model_name
import Helpers.config as conf

def preprocess_he(text):
    """
    Preprocess hebrew reports
    :param text: reports in hebrew
    :return: reports in hebrew without english words, numbers, punctuation, extra spaces...
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
    """Preprocess reports in english
    :param text: reports in english
    :return: reports in english without numbers, punctuation...
    """
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", " ", text)
    text = re.sub(r"(\n)", " ", text)
    text = re.sub(r"(\r)", "", text)
    text = re.sub(r' +', ' ', text, flags=re.I)
    return text


def rep_to_bigrams2(report, bigrams, clean_bigrams):
    """Replacing bigrams in the reports
    :param report: report text
    :param bigrams: bigram
    :param clean_bigrams: bigram with '_'
    :return:
    """
    for i, bigram in enumerate(clean_bigrams):
        report = report.replace(bigram, bigrams[i])
    return report

def get_bigram_ontology(real_file=conf.ontology_file, list_words_col=conf.expanded_words):
    """
    Retrieve all the bigrams in the expanded words of the file CXR ontology.csv
    :param real_file:
    :param list_words_col:
    :return:
    """
    original = pd.read_csv(real_file)
    original[list_words_col].fillna(' ', inplace=True)
    words = ','.join(original[list_words_col].tolist())
    output = []
    if not isinstance(words, (int, float)):
        words = words.split(',')
        for word in words:
            word = re.sub(r"[!\"#$%&'*+/<=>?@[\]^{|}~]", "", word)
            word = re.sub(r' +', ' ', word, flags=re.I)
            word = word.strip()
            word = word.lower()
            word = word.replace(' ', '_')
            if '_' in word:
                output.append(word)
        return output
    else:
        return []


def prepare_text(findings_cxr=conf.ontology_file, finding_name_col=conf.finding_col_ontology,
                 report_file=conf.reports_for_training, report_col=conf.col_report, source_col=conf.col_source,
                 lang='EN', **kwargs):
    """
    Prepare the text of the reports to train word2vec on it
    :param findings_cxr: File containing all the findings name
    :param finding_name_col: Name of the column containing the findings in this file
    :param report_file: File on what we want to train the model
    :param report_col: Name of the column report
    :param source_col: Name of the columns containing the source of the data
    :param lang: 'EN' or 'HE'
    :return: The reports preprocessed & the language of the text
    """
    data = pd.read_csv(report_file)
    findings_cxr = pd.read_csv(findings_cxr)
    analyzer = TfidfVectorizer(stop_words='english', **kwargs).build_analyzer()  # tokenization
    bigrams = []
    print('dealing with bigrams...')
    if lang == 'EN':
        data = data[data[source_col] != conf.HEBREW_REPORT_INDICATOR]
        findings = list(findings_cxr[finding_name_col])
        bigrams += get_bigram_ontology()
        bigrams += [ontology for ontology in findings if '_' in ontology]
        clean_bigrams = [bigram.replace(' ', '_') for bigram in bigrams]
        data[report_col] = data[report_col].apply(lambda x: rep_to_bigrams2(x, bigrams, clean_bigrams))
        print('Preprocess...')
        texts = data[report_col].apply(lambda s: preprocess_en(s)).apply(lambda s: analyzer(s))
        return texts
    elif lang == 'HE':
        data = data[data[source_col] == conf.HEBREW_REPORT_INDICATOR]
        bigrams += get_bigram_ontology(list_words_col=conf.finding_col_he)
        clean_bigrams = [bigram.replace(' ', '_') for bigram in bigrams]
        data[report_col] = data[report_col].apply(lambda x: rep_to_bigrams2(x, bigrams, clean_bigrams))
        print('Preprocess...')
        texts = data[report_col].apply(lambda s: preprocess_he(s)).apply(lambda s: analyzer(s))
        return texts


def train_word2vec(corpus_path, hebrew_model=False):
    """Train word2vec on the preprocessed reports
    :param corpus_path: path of the corpus we want to train on it
    :param hebrew_model: language of the training
    :return: The trained word2vec model name & entry point
    """
    lang = 'HE' if hebrew_model else 'EN'
    corpus_name = re.search(r'^/(.+/)*(.+)\.(.+)$', '/' + corpus_path).group(2)
    texts = prepare_text(report_file=corpus_path, lang=lang)
    bigrams = Phrases(texts, min_count=6)
    model_w2v = Word2Vec(bigrams[texts])
    total_examples = model_w2v.corpus_count
    print('Training...')
    model_w2v.train(bigrams[texts], total_examples=total_examples, epochs=conf.w2v_epochs)
    print('Saving model...')
    date = get_date_for_model_name()
    model_name = date + '_w2v_' + lang.lower() + '_' + corpus_name
    full_model_path = conf.MODELS_PATH + conf.TYPE_MODEL_PATH['word2vec'] + model_name
    model_w2v.save(full_model_path + '.model')
    print('The model has been saved under : ', full_model_path + '.model')
    return full_model_path + '.model'
