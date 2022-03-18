import torch
from transformers import BartTokenizer
from bart import UnifiedQABart, Bart
import pandas as pd
from tqdm import tqdm
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="./Drop/train.tsv")
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

    pattern = ['[Passage] [Question] <mask>',
                '[Passage] [Question] The answer is <mask>',
               '[Passage] According to the passage, [Question] <mask>',
               'Based on the following passage, [Question] <mask>. [Passage]'
               ]

    preds = prediction(model, tokenizer, qa_data, pattern[args.prompt_pattern], device)
    qa_data['pred'] = preds

    save_file_path = os.path.join(args.save_dir, 'result_pattern' + str(args.prompt_pattern) + '.csv')
    qa_data.to_csv(save_file_path, index=0)


if __name__ == '__main__':
    main()
