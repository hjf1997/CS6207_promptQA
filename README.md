# CS6207 Project: Soft-Prompt for open-form QA

## Requirement
Transformers package is required
```angular2html
pip install git+https://github.com/huggingface/transformers.git@7b75aa9fa55bee577e2c7403301ed31103125a35
```
## The datasets
Download pre-processed NarrativeQA dataset from [Google Cloud](https://console.cloud.google.com/storage/browser/unifiedqa/data/narrativeqa?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false).

## Pre-trained UnifiedQA-bart checkpoint
The BART models are downloaded from [this link](https://nlp.cs.washington.edu/ambigqa/models/unifiedQA/unifiedQA-bart.zip) (3.6G). Use `uncased` model for better performances.

