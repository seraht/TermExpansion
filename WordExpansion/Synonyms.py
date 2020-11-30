""""
Authors: ST & MM
Retrieving synonyms
"""

from WordExpansion.UMLS import *
from py_babelnet.calls import BabelnetAPI
from nltk.corpus import wordnet
import random
from WordExpansion.Filter_Sentences_Bert import *
import Helpers.config as cnf
import transformers
import fasttext
from gensim.models import Word2Vec
import nltk


def get_word2vec(finding, model_path, corpus=None, top_n=10, **kwargs):
    """
    We use the Word2Vec algorithm to find the topn synonyms, then we filter the synonyms by the POS_TAG
    We want only the synonyms that have the same POS_TAG as the finding
    :param finding: (str) Finding name to look for
    :param model_path: (str) Pretrained Word2Vec model path
    :param corpus: Corpus path to get the frequency of the given term
    :param top_n: (int) Number of synonyms to return
    :return: The list of the synonyms with their corresponding score
    """
    output = []
    word = finding.replace(' ', '_')
    model = Word2Vec.load(model_path)
    try:
        result = model.wv.most_similar(word, topn=top_n, **kwargs)
        if corpus is None:
            return result
        for token in result:
            output.append((token[0], round(token[1], 3), word_frequency(token[0], corpus)))
        return output
    except Exception as e:
        return [f'Error when using Word2Vec algorithm: \n \
                ------------------------------------- \n {e}']


def get_fasttext(finding, model_path, corpus=None, top_n=10, **kwargs):
    """
    We use the FastText algorithm to find the top_n synonyms
    :param finding: (str) Finding name to look for
    :param model_path: (str) Pretrained FastText model path
    :param corpus: Corpus path to get the frequency of the given term
    :param top_n: (int) Number of synonyms to return
    :return: The list of the synonyms with their corresponding score
    """
    output = []
    word = finding.replace(' ', '_')
    model = fasttext.load_model(model_path)
    result = model.get_nearest_neighbors(word, top_n, **kwargs)
    if corpus is None:
        return result
    for token in result:
        output.append((token[1], round(token[0], 3), word_frequency(token[1], corpus)))
    return output


def get_wordnet(word, top_n=10):
    """
    We use WordNet to get the synonyms of a given word
    :param word: (str) Finding
    :param top_n: (int) Number of synonyms to return
    :return: The list of synonyms
    """
    nltk.download('wordnet')
    synonyms = []
    try:
        wordsyn = wordnet.synsets(word)
        for syn in wordsyn:
            for lem in syn.lemmas():
                if lem.name() not in synonyms and lem.name() != word:
                    term = wordnet.synsets(lem.name())[0]
                    synonyms.append((lem.name(), term.wup_similarity(wordsyn[0])))
        if synonyms:
            synonyms = sorted(synonyms, key=lambda x: x[1], reverse=True)
            return synonyms[:top_n]
        else:
            return []

    except Exception as e:
        return f'Error when using WordNet algorithm : \n  \
                 ------------------------------------- \n  {e}'


def get_umls(finding, key=cnf.UMLS_rachel_api, top_n=10):
    """
    Retrieve synonyms of an input word by using the UMLS API
    :param finding: (str) Finding
    :param key: (str) The API key I found on Rachel UMLS account
    :param top_n: Number of synonyms
    :return: The list of synonyms
    """
    return get_synonym(key, finding)[:top_n]


def get_babelnet(finding, babelnet_key=cnf.babelnet_key, top_n=10, hebrew_api=False):
    """
    Retrieve synonyms of an input word by using BabelNet API
    :param finding: (str) Finding name, term to look for
    :param babelnet_key: (str) BabelNet API key is limited to 1000 request per month,
     to get more information go to https://babelnet.org/guide
    :param top_n: Number of synonyms
    :param hebrew_api: (bool) The language of the synonyms
    :return: The list of synonyms
    """
    api = BabelnetAPI(babelnet_key)
    lang = 'HE' if hebrew_api else 'EN'
    try:
        senses = api.get_senses(lemma=finding, searchLang=lang)
        syn_h = []
        for i in range(len(senses)):
            syn_h.append(senses[i]['properties']['fullLemma'])

        return list(set(syn_h))[:top_n]
    except Exception as e:
        return f'Error when using BabelNet algorithm : \n  \
                 ------------------------------------- \n  {e}'

def get_bert(finding, model_path, corpus=cnf.REPORTS_SAMPLE_PATH, top_n=10, hebrew_model=False):
    """
    Retrieve synonyms of an finding name by calling a Bert model
    :param finding: Finding name, term to look for
    :param model_path: model path of the model
    :param corpus: Corpus to get word frequency
    :param top_n: Number of synonyms to return
    :param hebrew_model: language of the model
    :return: The list of synonyms with frequencies in the corpus
    """
    result = []
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    fill_mask = transformers.pipeline('fill-mask', model=model, tokenizer=tokenizer, topk=top_n)
    try:
        sentence = random.choice(select_sentences_for_finding(corpus, finding, hebrew_model))
    except IndexError:
        return ["Word not found in the corpus"]

    sentence = sentence.replace(finding, fill_mask.tokenizer.mask_token)
    mask_output = fill_mask(sentence)

    for mask in mask_output:
        if mask['token_str'] != finding:
            result.append((mask['token_str'], mask['score']))

    output = []
    for token in result:
        output.append((token[0], round(token[1], 3), word_frequency(token[0], corpus)))

    return output


def expand_term_api(api_name, finding, corpus=cnf.REPORTS_SAMPLE_PATH, top_n=10, hebrew_api=False):
    """
    Get synonyms of a finding name by using an api
    :param api_name: name of the api
    :param finding: Finding name, term to look for
    :param corpus: corpus name to get the frequency
    :param top_n: Number of synonyms to return
    :param hebrew_api: language of the api (only for babelnet)
    :return:
    """
    output = []
    if api_name.lower() == 'babelnet':
        result = get_babelnet(finding, top_n=top_n, hebrew_api=hebrew_api)
        for token in result:
            output.append((token, word_frequency(token, corpus)))

    elif api_name.lower() == 'wordnet':
        result = get_wordnet(finding, top_n=top_n)
        if len(result) < 1:
            output = ['No synonym for the word ' + finding]
        else:
            for token in result:
                output.append((token[0], token[1], word_frequency(token[0], corpus)))

    elif api_name.lower() == 'umls':
        result = get_umls(finding, top_n=top_n)
        for token in result:
            output.append((token, word_frequency(token.lower(), corpus)))

    return output

def word_frequency(finding, corpus):
    """Retrieve the word occurence of a given word in a corpus
    :param finding: Finding Name, term to look foor
    :param corpus: dataframe containing a column report
    :return: number of occurence
    """
    corpus = pd.read_csv(corpus)
    data = corpus[cnf.col_report].tolist()
    word = finding.replace('_', ' ').lower()
    length = len(word)
    return sum(element[index:index + length].lower() == word for element in data for index, char in enumerate(element))
