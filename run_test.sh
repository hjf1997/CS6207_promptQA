python3 prompt_tuning.py \
--do_predict \
--fix_LM \
--randomize_prompt \
--output_dir checkpoint/NarrativeQA_soft \
--predict_file NarrativeQA/test.tsv \
--predict_batch_size 32 \
--append_another_bos \
--gpu_ids 0 \
--pattern_id 0