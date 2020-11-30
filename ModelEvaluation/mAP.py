"""
Authors: Serah Tapia & Michael Marcus
mAP logic evaluation
"""

from WordExpansion.Synonyms import *
import statistics
import Helpers.config as conf


def calc_map(retrieved_list, gold_standard_list):
    """
    Calculate the map between the retrieved list and the gold standard list
    :param retrieved_list: Synonyms list
    :param gold_standard_list: Gold standard synonyms list
    :return: Score of the map according to the GPT
    """
    rel = 0
    precision = []
    for i in range(len(retrieved_list)):
        if retrieved_list[i].lower() in gold_standard_list:
            rel += 1
            precision.append(rel / (i + 1))
    if len(retrieved_list) > 0:
        result = sum(precision) / len(retrieved_list)
        return round(result, 3)
    else:
        return 0


def get_gold_standard(finding, gold_file=conf.gold_standard_list_file, list_words_col=conf.synonyms_col,
                      finding_name_col=conf.finding_col):
    """
    Retrieve all the synonyms of a finding in the gold standard list
    :param finding: Term to look for
    :param gold_file: The name of the file with the gold standard list
    :param list_words_col: The name of the columns with the synonyms
    :param finding_name_col: The name of the columns with the findings
    :return: The list of the synonyms for the given finding
    """
    finding = finding.replace(' ', '_')
    original = pd.read_excel(gold_file)
    original = original[original[finding_name_col] == finding]
    words = original[list_words_col].values
    if len(words) == 0:
        print(finding + ' has no synonym in the gold standard list')
    return words


def get_synonyms_list(finding, model_type, model_path=None, corpus=None, hebrew_model=False):
    """
    Retrieve all the synonyms of a given finding using an algorithm
    :param finding: Term to look for
    :param model_type: Type of the model it can be an api name as well
    :param model_path: Path of the algorithm (for word2vec, fasttext, bert)
    :param corpus: corpus only for bert models
    :param hebrew_model: Boolean declaring if the model is in hebrew or not
    :return: List of synonyms using the given algorithm
    """
    if model_type == 'word2vec':
        synonyms = get_word2vec(finding, model_path=model_path)
        if 'Error when using Word2Vec algorithm:' in synonyms:
            return []
        else:
            return [i[0].replace('_', ' ') for i in synonyms]
    elif model_type == 'fasttext':
        synonyms = get_fasttext(finding, model_path)
        return [i[1].replace('_', ' ') for i in synonyms]
    elif model_type == 'wordnet':
        synonyms = get_wordnet(finding)
        return [i[0].replace('_', ' ') for i in synonyms]
    elif model_type == 'babelnet':
        synonyms = get_babelnet(finding, hebrew_api=hebrew_model)
        return [i.replace('_', ' ') for i in synonyms]
    elif model_type == 'umls':
        synonyms = get_umls(finding)
        return [i.replace('_', ' ') for i in synonyms]
    elif model_type == 'bert':
        synonyms = get_bert(finding, model_path, corpus=corpus, hebrew_model=hebrew_model)
        return [i[0].replace('_', ' ') for i in synonyms]
    else:
        raise NameError('Error when choosing your model name')


def evaluate_term_map(finding, model_type, model_path=None, corpus=None, hebrew_model=False):
    """
    Evaluates the mAP score of a finding term
    :param finding: Term to look for
    :param model_type: Type of the model it can be an api name as well
    :param model_path: Path of the algorithm (for word2vec, fasttext, bert)
    :param corpus: corpus only for bert models
    :param hebrew_model: Boolean declaring if the model is in hebrew or not
    :return: mAP score of the term on a given gold standard list
    """
    gold_list = get_gold_standard(finding)
    return calc_map(get_synonyms_list(finding, model_type, model_path, corpus, hebrew_model), gold_list)


def evaluate_algorithm(model_type, model_path=None, corpus=None, hebrew_model=False,
                       gold_file=conf.gold_standard_list_file):
    """
    Evaluates the algorithm by using a list of terms with in gold standard list
    :param model_type: Type of the model it can be an api name as well name
    :param model_path: Path of the algorithm (for word2vec, fasttext, bert)
    :param corpus: corpus only for bert models
    :param hebrew_model: Boolean declaring if the model is in hebrew or not
    :param gold_file: Gold Standard List
    :return: mAP scores (mean & median) on the given terms in the GSL
    """
    df = pd.read_excel(gold_file)
    terms_list = df[conf.finding_col].values
    terms_list = list(set(terms_list))
    summation = []
    for finding in terms_list:
        summation.append(evaluate_term_map(finding, model_type, model_path, corpus, hebrew_model))
    map_score = round(sum(summation) / len(summation), 3)
    median_map_score = round(statistics.median(summation), 3)
    return map_score, median_map_score


def evaluate_all_algorithms(w2v_model, fsttxt_model, bert_model, corpus=conf.REPORTS_SAMPLE_PATH):
    """
    Evaluates all the synonyms of the golden list for each algorithm
    :param w2v_model: W2V model to use (Word2vecModel)
    :param fsttxt_model: FastText model to use (FasttextModel)
    :param bert_model: Bert model to use (BertModel)
    :param bert_model:
    :param corpus: The necessary corpus for Bert
    :return: DataFrame
    """

    algorithms = conf.algorithms
    for algo in algorithms:
        if algo == 'word2vec':
            result = w2v_model.evaluate_algorithm()
            print(f'Average Score for W2V: \n '
                  f'------------------------------ \n'
                  f'{round(result[0], 3)}')

            print(f'Median Score for W2V: \n '
                  f'------------------------------ \n '
                  f'{round(result[1], 3)}\n')

        elif algo == 'fasttext':
            result = fsttxt_model.evaluate_algorithm()
            print(f'Average Score for FastText: \n '
                  f'------------------------------ \n'
                  f'{round(result[0], 3)}')
            print(f'Median Score for Fastext: \n '
                  f'------------------------------ \n '
                  f'{round(result[1], 3)}\n')

        elif algo == 'bert':
            result = bert_model.evaluate_algorithm(corpus=corpus)
            print(f'Average Score for Bert: \n '
                  f'------------------------------ \n'
                  f'{round(result[0], 3)}')
            print(f'Median Score for Bert: \n '
                  f'------------------------------ \n '
                  f'{round(result[1], 3)}\n')

        else:
            result = evaluate_algorithm(algo)
            print(f'Average Score for {algo.title()}: \n '
                  f'------------------------------ \n '

                  f'{round(result[0], 3)}')
            print(f'Median Score for {algo.title()}: \n '
                  f'------------------------------ \n'
                  f'{round(result[1], 3)}\n')
