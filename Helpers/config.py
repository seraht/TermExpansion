
# Paths
MODELS_PATH = "Models/"
MODEL_TYPE = {
    'word2vec': 'word2vec',
    'fasttext': 'fasttext',
    'bert': 'bert'
}
TYPE_MODEL_PATH = {
    'word2vec': MODEL_TYPE['word2vec'] + '/',
    'fasttext': MODEL_TYPE['fasttext'] + '/',
    'bert': MODEL_TYPE['bert'] + '/'
}
DATA_PATH = "TrainingModels/data/"
REPORTS_SAMPLE_PATH = 'TrainingModels/data/sample_for_ITC.csv'
ARCHIVES_PATH = 'Archives/'

# General
finding_col_he = "heb_terms_list"
ontology_file = "TrainingModels/data/CXR_ontology_final.csv"
HEBREW_REPORT_INDICATOR = 'clalit'
HEBREW_IDENTIFIER_NAME = "hebrew"
ENGLISH_IDENTIFIER_NAME = "english"

# expand_to_sheet
columns_expand_to_csv = ['Finding_name', 'Synonyms']
EXPORT_SHEET_PATH = 'ExportExcel/SynonymsSheets/'

# mAP
gold_standard_list_file = 'Helpers/GoldStandardList.xlsx'
synonyms_col = 'Synonyms'
finding_col = 'finding_name'
algorithms = ['word2vec', 'fasttext', 'wordnet', 'umls', 'bert', 'babelnet']

# Training_Word2Vec, Fasttext
expanded_words = 'eng_terms_list'
finding_col_ontology = 'finding_name'
reports_for_training = 'TrainingModels/data/chest_xray_reports.csv'
w2v_epochs = 5

# Training Bert
bert_model = {
    'english': 'allenai/scibert_scivocab_uncased',
    'hebrew': 'bert-base-multilingual-uncased'
}

col_report = 'report'
col_source = 'data_source'


# Filter_sentences_bert
negation = [" no ", "n't ", "not", "לא", "אין"]

# Synonyms
UMLS_rachel_api = 'c8352956-d42f-4369-bd81-5936c34a3994'
# You can recreate the babelnet key by creating a new account with a different email if you exceed the limit
babelnet_key = "f44d116b-399d-4dab-848e-f685e976e7f6"


# CLI menu
actions = "- Train : 1 \n" \
          "- Expand Term : 2 \n" \
          "- Evaluate Term with mAP : 3\n" \
          "- Evaluate Algorithm with mAP : 4 \n" \
          "- Compare Algorithms with mAP : 5 \n" \
          "- Output Expanded Term into CSV : 6 \n" \
          "- List all the Pretrained Models : 7 \n "

training_algo = "- FastText : 1 \n" \
          "- Word2Vec: 2 \n" \
          "- Bert : 3 \n"

algos = "- FastText : 1 \n" \
        "- Word2Vec: 2 \n" \
        "- Bert : 3 \n" \
        "- Umls : 4 \n" \
        "- WordNet : 5 \n" \
        "- Babelnet : 6 \n"
