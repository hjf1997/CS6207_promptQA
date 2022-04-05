import os
import argparse
import logging

import random
import numpy as np
import torch
from run import run


def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--train_file", default="NarrativeQA/train.tsv")
    parser.add_argument("--predict_file", default="NarrativeQA/dev.tsv")
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--skip_inference", action='store_true')
    # parser.add_argument("--pattern", type=int, default=0)
    parser.add_argument("--encoder_pattern", type=int, default=0)
    parser.add_argument("--decoder_pattern", type=int, default=0)
    parser.add_argument("--num_few_shot", type=int, default=0)

    ## Model parameters
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument('--checkpoint_step', type=int, default=0)
    parser.add_argument("--do_lowercase", action='store_true')
    parser.add_argument("--gpu_ids", type=str, default='-1')  # -1 means cpu

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--max_output_length', type=int, default=100)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--append_another_bos", action='store_true', default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=400, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10)

    # Other parameters
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=2400,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # patterns = ['[Passage] [Question]',
    #             '[Passage] [Question] <mask>',
    #             '[Passage] [Question] The answer is <mask>',
    #            '[Passage] According to the above passage, [Question] <mask>',
    #            'Based on the following passage, [Question] <mask>. [Passage]',
    #            '[Question] [Passage]',
    #            '[Question] <mask> [Passage]',
    #            ]
    # args.pattern = patterns[args.pattern]

    encoder_patterns = ['[Passage] [Question]',
                '[Passage] [Question] <mask>',
                '[Passage] [Question] The answer is <mask>',
               '[Passage] According to the above passage, [Question] <mask>',
               'Based on the following passage, [Question] <mask>. [Passage]',
               '[Question] [Passage]',
               '[Question] <mask> [Passage]',]

    decoder_patterns = ['[Answer]',
                        'The answer is [Answer]',
                        '[Question] The answer is [Answer]',
                        '[Question] [Answer]']

    args.encoder_pattern = encoder_patterns[args.encoder_pattern]
    args.decoder_pattern = decoder_patterns[args.decoder_pattern]

    ##### Start writing logs

    log_filename = "{}log.txt".format("" if args.do_train else "eval_")

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError("If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    logger.info("Using {} gpus".format(len(args.gpu_ids)))
    run(args, logger)

if __name__ == '__main__':
    main()