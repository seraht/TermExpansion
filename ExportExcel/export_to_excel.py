"""
Authors: ST & MM
Expanding a term into an excel file
"""

from WordExpansion.Synonyms import *
import Helpers.config as conf
import Helpers.helpers as help

def output_expanded_term_to_excel(term, bert_model, w2v_model, fstxt_model, corpus=cnf.REPORTS_SAMPLE_PATH, top_n=10):
    """
    Retrieve all the synonyms of a term in a excel file
    :param term: Given finding (str)
    :param w2v_model: W2V model to use (Word2vecModel)
    :param fstxt_model: FastText model to use (FasttextModel)
    :param bert_model: Bert model to use (BertModel)
    :param bert_model:
    :param corpus: The necessary corpus for Bert
    :param top_n: Number of synonyms
    :return: DataFrame
    """
    synonyms_w2v = w2v_model.expand_terms(term)
    if 'Error when using Word2Vec algorithm:' in synonyms_w2v[0]:
        word2vec = []
    else:
        word2vec = [i[0].replace('_', ' ').lower() for i in synonyms_w2v]

    synonyms_bbnet_en = get_babelnet(term, top_n=top_n)
    babelnet_en = [i.replace('_', ' ').lower() for i in synonyms_bbnet_en]

    synonyms_bbnet_he = get_babelnet(term, hebrew_api=True)
    babelnet_he = [i.replace('_', ' ').lower() for i in synonyms_bbnet_he]

    synonyms_bert = bert_model.expand_terms(term, corpus_path=corpus)
    bert = [i[0].replace('_', ' ').lower() for i in synonyms_bert]

    synonyms_ft = fstxt_model.expand_terms(term)
    fstxt = [i[0].replace('_', ' ').lower() for i in synonyms_ft]

    synonyms_wnet = get_wordnet(term, top_n=top_n)
    wordnet_syn = [i[0].replace('_', ' ').lower() for i in synonyms_wnet]

    synonyms_umls = get_umls(term)
    umls = [i.replace('_', ' ').lower() for i in synonyms_umls]

    synonyms = list(set(word2vec + fstxt + wordnet_syn + babelnet_en + babelnet_he + umls + bert))

    df_syn = pd.DataFrame()
    df_syn[cnf.columns_expand_to_csv[1]] = synonyms
    df_syn[cnf.columns_expand_to_csv[0]] = term
    df_syn = df_syn[[cnf.columns_expand_to_csv[0], cnf.columns_expand_to_csv[1]]]
    df_syn = df_syn.reset_index(drop=True)
    date = help.get_date_for_model_name()
    sheet_path = conf.EXPORT_SHEET_PATH + f"{term}_synonyms_{date}.xlsx"
    df_syn.to_excel(sheet_path)

    print("The csv file has been saved in ", sheet_path)

    return df_syn
