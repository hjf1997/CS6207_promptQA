import torch
from transformers import BartTokenizer
from bart import UnifiedQABart, Bart
import pandas as pd
from tqdm import tqdm
import argparse
import os


def prediction(model, tokenizer, qa_data, pattern: str, device):
    preds = []
    for index, row in tqdm(qa_data.iterrows(), desc='Testing', total=len(qa_data)):
        question, passage = transform_data(row['text'])
        # input_text = pattern.replace('[Question]', question + '\\n')
        # input_text = input_text.replace('[Passage]', passage + '\\n')
        input_text = pattern.replace('[Question]', question)
        input_text = input_text.replace('[Passage]', passage)
        pred = model.generate_from_string(input_text, tokenizer=tokenizer, device=device)
        preds.append(pred[0])
    return preds


def transform_data(sen: str):
    """
    Transform training data into question and passage
    """
    sen = sen.split('\\n')
    question, passage = sen
    return question, passage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='./NarrativeQE/data_narrativeqa_test.tsv')
    parser.add_argument("--save_dir", type=str, default='./NarrativeQE/results')
    parser.add_argument("--prompt_pattern", type=int, default=0)
    parser.add_argument("--gpu_ids", type=int, default='-1')  # -1 means cpu
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu_ids) if args.gpu_ids != -1 else "cpu")
    base_model = "facebook/bart-large"
    unifiedqa_path = "unifiedQA-bart/unifiedQA-uncased/best-model.pt"  # path to the downloaded checkpoint

    tokenizer = BartTokenizer.from_pretrained(base_model)
    # model = Bart.from_pretrained(base_model, state_dict=torch.load(unifiedqa_path))
    model = Bart.from_pretrained(base_model)
    model = model.to(device)
    model.eval()

    qa_data = pd.read_csv(args.dataset_path, sep='\t', header=None)
    qa_data.columns = ['text', 'ans']

    pattern = ['[Question] [Passage] <mask>',
               '[Passage] According to the passage, [Question] <mask>',
               'Based on the following passage, [Question] <mask>. [Passage]'
               ]

    preds = prediction(model, tokenizer, qa_data, pattern[args.prompt_pattern], device)
    qa_data['pred'] = preds

    save_file_path = os.path.join(args.save_dir, 'result_pattern' + str(args.prompt_pattern) + '.csv')
    qa_data.to_csv(save_file_path, index=0)


if __name__ == '__main__':
    main()
