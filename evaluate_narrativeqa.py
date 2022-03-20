import numpy as np
import rouge
import pandas as pd
from tqdm import tqdm

rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
)

def rouge_l(p, g):
    return rouge_l_evaluator.get_scores(p, g)


if __name__ == '__main__':
    file_path = './checkpoint/NarrativeQA/predictions.csv'
    qa_results = pd.read_csv(file_path)
    scores_r = []
    scores_p = []
    scores_f = []
    for index, row in tqdm(qa_results.iterrows(), desc='Evaluating', total=len(qa_results)):
        gt, pred = row['answer'], row['prediction']
        # if not isinstance(pred, str) or pred == '':
        #     continue
        rouge_l_score = rouge_l(pred, gt)
        scores_r.append(rouge_l_score[0]['rouge-l']['r'])
        scores_p.append(rouge_l_score[0]['rouge-l']['p'])
        scores_f.append(rouge_l_score[0]['rouge-l']['f'])
    scores_r, scores_p, scores_f = np.array(scores_r), np.array(scores_p), np.array(scores_f)
    scores_r, scores_p, scores_f = scores_r.reshape([-1, 2]), scores_p.reshape([-1, 2]), scores_f.reshape([-1, 2])
    scores_r, scores_p, scores_f = np.max(scores_r, axis=-1), np.max(scores_p, axis=-1), np.max(scores_f, axis=-1)
    scores_r, scores_p, scores_f = scores_r.mean(), scores_p.mean(), scores_f.mean()

    print('score_r %f'% scores_r)
    print('score_p %f'% scores_p)
    print('score_f %f'% scores_f)
