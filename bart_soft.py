import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

class QABart_prompt(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.emb_dim = self.model.shared.embedding_dim
        self.mid_dim = 2*self.emb_dim

        # initialize soft prompt
        self.prompt_len = 10 # hyper

        self.wte = nn.Embedding(self.prompt_len, self.emb_dim)
        self.soft_mlp = nn.Sequential(
            nn.Linear(self.emb_dim, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.emb_dim)
        )

        # self.soft_prompt = self.randomize_prompt()

    # def randomize_prompt(self):
    #     # mean, std = self.model.shared.weight.mean(0), self.model.shared.weight.std(0)
    #     soft_prompt = torch.zeros([self.prompt_len, self.emb_dim]).cuda()
    #     for i in range(self.prompt_len):
    #         soft_prompt[i] = torch.normal(mean, std)
    #     soft_prompt = soft_prompt.detach().clone()

    #     return soft_prompt
    
    def get_soft_prompt(self):
        prompt_tokens = torch.arange(self.prompt_len).long().cuda()
        soft_prompt = self.wte(prompt_tokens)
        soft_prompt = self.soft_mlp(soft_prompt)

        return soft_prompt

    def get_input_embeds(self, input_ids, attention_mask, random=False):
        input_shape = input_ids.shape
        inputs_embeds = self.model.shared(input_ids)
        
        soft_prompt = self.get_soft_prompt()
        prefix_prompt = self.soft_mlp(soft_prompt).unsqueeze(0).repeat(input_shape[0], 1, 1)
        inputs_embeds = torch.cat([prefix_prompt, inputs_embeds[:, :-self.prompt_len, :]], dim=1)

        padding_attention_mask = torch.ones([input_shape[0], self.prompt_len]).cuda()
        attention_mask = torch.cat([padding_attention_mask, attention_mask[:, :-self.prompt_len]], dim=-1)

        return inputs_embeds, attention_mask
    
    def forward(self, input_ids, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None,
            use_cache=False, is_training=False, inputs_embeds=None, random=True, **model_inputs):

        if input_ids is not None:
            inputs_embeds, attention_mask = self.get_input_embeds(input_ids, attention_mask, random)

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
        else:
            _decoder_input_ids = decoder_input_ids.clone()
        
        decoder_inputs_embeds = self.model.shared(decoder_input_ids)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=use_cache,
            **model_inputs
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        if is_training:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                              decoder_input_ids.view(-1))
            loss = torch.sum(losses * decoder_attention_mask.float().view(-1))
            return loss
        return Seq2SeqLMOutput(
            loss=0,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate_from_string(self, _input, tokenizer=None, device='cpu', **generator_args):
        assert tokenizer is not None
        if isinstance(_input, str):
            _input = [[0] + tokenizer.encode(_input)]
        if isinstance(_input, list) and isinstance(_input[0], str):
            _input = [[0] + tokenizer.encode(i) for i in _input]
        if isinstance(_input, list):
            if isinstance(_input[0], int):
                _input = [_input]
            _input = torch.LongTensor(_input).to(device)
        if _input.shape[1] > 1024:  # maximum words constrain
            return ['']
        res = self.generate(_input, **generator_args)
        return ([tokenizer.decode(x, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True).strip() for x in res])



