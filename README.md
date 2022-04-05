# CS6207 Project: Enhancing Abstractive Question Answering with Prompts: A Pilot Study

### Junfeng Hu, Hongfu Liu, Xu Liu

## Requirement
Transformers package is required
```angular2html
pip install transformers
```
## The datasets
Download pre-processed NarrativeQA dataset from [Google Cloud](https://console.cloud.google.com/storage/browser/unifiedqa/data/narrativeqa?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false).

## Fixed Prompt

### Patterns

| index | Encoder Prompt                                                 | Decoder Prompt                     |
|------|----------------------------------------------------------------|--------------------------------------|
| 0    | [Passage] [Question]                                           | [Answer]         |
| 1    | [Passage] [Question] \<mask\>                                  | The answer is [Answer]          |
| 3    | [Passage] [Question] The answer is \<mask\>                    |            |
| 4    | [Passage] According to the above passage, [Question] \<mask\>  |             |
| 5    | Based on the following passage, [Question] \<mask\>. [Passage] |            |
| 6    | [Question] [Passage]                                           |  |
| 7    | [Question] \<mask\> [Passage]                                  |  |


### Training
```angular2html
python fixed_main.py --do_train\
--output_dir checkpoint/NarrativeQA_enp1_decp0_128\
--train_file NarrativeQA/train.tsv\
--predict_file NarrativeQA/dev.tsv\
--train_batch_size 32\
--predict_batch_size 64\
--append_another_bos\
--do_lowercase\
--gpu_ids 3\
--eval_period 100\
--verbose\
--num_few_shot 128\
--encoder_pattern 1\
--decoder_pattern 0\
```

### Evaluation
```angular2html
python fixed_main.py --do_predict\
--output_dir checkpoint/NarrativeQA_enp1_decp0_128\
--predict_file NarrativeQA/test.tsv\
--predict_batch_size 64\
--append_another_bos\
--gpu_ids 3\
--encoder_pattern 1\
--decoder_pattern 0\
--verbose
```
## Soft Prompt

### Training
```
python soft_main.py --do_train\
--train_file NarrativeQA/train.tsv\
--predict_file NarrativeQA/dev.tsv\
--append_another_bos\
--do_lowercase --verbose\
--pattern_id 0\
--predict_batch_size 32\
--num_few_shot 32\
--train_batch_size 28\
--prompt_len 1\
--eval_period 50\
--gpu_ids 3\
--output_dir checkpoint/num32_bs28_len1_random\
--randomize_prompt
```

### Evaluation
```angular2html
python soft_main.py --do_predict\
--predict_file NarrativeQA/test.tsv\
--predict_batch_size 32\
--append_another_bos\
--pattern_id 0\
--verbose\
--gpu_ids 3\
--prompt_len 1\
--output_dir checkpoint/num32_bs28_len1_random\
--randomize_prompt
```

## Calculate ROUGE-L Metric
```angular2html
python evaluate_narrativeqa.py --file_path ./checkpoint/NarrativeQA_enp1_decp0_128/predictions.csv
```
