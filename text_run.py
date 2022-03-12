from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
TXT = "My friends are <mask> but they eat too many carbs."

model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
output = model(input_ids)
logits = output.logits

masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)

tokenizer.decode(predictions).split()