#term_expansion_api
**An api to call the different term expansion algorithms**

## API Usage
algo_name = {fasttext, word2vec, bert}

1) Training Algorithm: `train_algo_api.py`
- `python -W ignore train_algo_api.py --algo algo_name`
2) Expand Term: `expand_term_api.py`
- `python -W ignore expand_term_api.py --algo algo_name --finding finding_term --algo_path path_model`
3) Export Term to csv: `export_term_csv_api.py`
- `python -W ignore export_term_csv_api.py --fasttext path_ft_model --word2vec path_w2v_model --bert path_bert_model --finding finding_term`
4) Evaluate Algorithm with mAP: `algorithm_evaluation_api.py`
- `python -W ignore algorithm_evaluation_api.py --algo algo_name --finding finding_term --algo_path path_model`
- `python -W ignore algorithm_evaluation_api.py -A`



## Comments

 - The trained models are deliberately not in the repo.
 - You can find them under algo/users/michael/Nlp_project/api/zebra_term_expansion_api/Models/ in Zebra's server and move them as they are in the local Models/ folder
 

