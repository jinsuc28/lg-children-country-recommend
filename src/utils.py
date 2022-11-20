import numpy as np

def _ndcg_calculator(gt, rec, idcg):
    dcg = 0.0
    for i, r in enumerate(rec):
        if r in gt:
            dcg += 1.0 / np.log(i + 2)
    return dcg / idcg

def ndcg_calculator(answer, submission, n):
    idcg = sum((1.0 / np.log(i + 1) for i in range(1, n + 1)))

    assert (answer.profile_id != submission.profile_id).sum() == 0

    ndcg_list = []
    for (_, row_answer), (_, row_submit) in zip(answer.iterrows(), submission.iterrows()):
        ndcg_list.append(_ndcg_calculator(row_answer.album_id, row_submit.album_id, idcg))

    ndcg_score = sum(ndcg_list) / len(answer)
    return ndcg_score