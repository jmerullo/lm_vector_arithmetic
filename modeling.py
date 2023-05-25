import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BloomTokenizerFast 
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

def load_gptj():
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision='float16', torch_dtype=torch.float16).to(device)
    return model, tokenizer

def load_gpt2xl():
    device= get_device()
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch.float16).to(device)
    return model, tokenizer

def load_gpt2(version):
    device= get_device()
    tokenizer = AutoTokenizer.from_pretrained(version)
    model = AutoModelForCausalLM.from_pretrained(version, torch_dtype=torch.float16).to(device)
    return model, tokenizer

def load_bloom(version):
    #device = get_device()
    print("loading bloom...")
    tokenizer = AutoTokenizer.from_pretrained(version)
    #max_memory = {0:'12GIB', 1:'16GIB', 2:'16GIB', 'cpu':"400GIB"}#, offload_dir = 'offload'
    #model = AutoModelForCausalLM.from_pretrained(version, max_memory=max_memory, device_map='auto', torch_dtype=torch.bfloat16).eval()
    
    model = AutoModelForCausalLM.from_pretrained(version, device_map='auto', torch_dtype=torch.bfloat16).eval()
    return model, tokenizer

def load_bloom_petals(version):
    from petals import DistributedBloomForCausalLM
    device= get_device()
    tokenizer = BloomTokenizerFast.from_pretrained(version)
    model = DistributedBloomForCausalLM.from_pretrained(version).eval().to(device)
    return model, tokenizer

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class BloomIdentityLayer(nn.Module):
    def __init__(self):
        super(BloomIdentityLayer, self).__init__()
    def forward(self, x, y):
        return x+y #bloom expects the MLP to handle the residual connection


class ModelWrapper(nn.Module):

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model.eval()
        self.model.activations_ = {}
        self.tokenizer = tokenizer
        self.device = get_device()
        self.num_layers = len(self.model.transformer.h)
        self.hooks  = []
        self.layer_pasts = {}

    def tokenize(self, s):
        return self.tokenizer.encode(s, return_tensors='pt').to(self.device)

    def list_decode(self, inpids):
        return [self.tokenizer.decode(s) for s in inpids]

    def layer_decode(self, hidden_states):
        raise Exception("Layer decode has to be implemented!")

    def get_layers(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states, true_logits = outputs.hidden_states, outputs.logits
        logits = self.layer_decode(hidden_states)
        #logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1) #we used to just replace the last logits because we were applying ln_f twice
        return torch.stack(logits).squeeze(-1)#, true_logits.squeeze(0)

    def get_layers_w_attns(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, output_attentions=True, **kwargs)
        hidden_states, true_logits = outputs.hidden_states, outputs.logits
        logits = self.layer_decode(hidden_states)
        #logits[-1] = true_logits.squeeze(0)[-1].unsqueeze(-1)
        return torch.stack(logits).squeeze(-1), outputs.attentions#, true_logits.squeeze(0)

    def rr_per_layer(self, logits, answer, debug=False):
        #reciprocal rank of the answer at each layer
        answer_id = self.tokenizer.encode(answer)[0]
        if debug:
            print("Answer id", answer_id, answer)

        rrs = []
        for i,layer in enumerate(logits):
            soft = F.softmax(layer,dim=-1)
            sorted_probs = soft.argsort(descending=True)
            rank = float(np.where(sorted_probs.cpu().numpy()==answer_id)[0][0])
            rrs.append(1/(rank+1))

        return np.array(rrs)

    def prob_of_answer(self, logits, answer, debug=False):
        answer_id = self.tokenizer.encode(answer)[0]
        if debug:
            print("Answer id", answer_id, answer)
        answer_probs = []
        first_top = -1
        mrrs = []
        for i,layer in enumerate(logits):
            soft = F.softmax(layer,dim=-1)
            answer_prob = soft[answer_id].item()
            sorted_probs = soft.argsort(descending=True)
            if debug:
                print(f"{i}::", answer_prob)
            answer_probs.append(answer_prob)
        #is_top_at_end = sorted_probs[0] == answer_id
        return np.array(answer_probs)

    def print_top(self, logits, k=10):
        for i,layer in enumerate(logits):
            print(f"{i}", self.tokenizer.decode(F.softmax(layer,dim=-1).argsort(descending=True)[:k]) )

    def topk_per_layer(self, logits, k=10):
        topk = []
        for i,layer in enumerate(logits):
            topk.append([self.tokenizer.decode(s) for s in F.softmax(layer,dim=-1).argsort(descending=True)[:k]])
        return topk

    def get_activation(self, name):
        #https://github.com/mega002/lm-debugger/blob/01ba7413b3c671af08bc1c315e9cc64f9f4abee2/flask_server/req_res_oop.py#L57
        def hook(module, input, output):
            if "in_sln" in name:
                num_tokens = list(input[0].size())[1]
                self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    num_tokens = list(output[0].size())[1]
                    self.model.activations_[name] = output[0][:, num_tokens - 1].detach()
                    self.model.activations_['in_'+name] = input[0][:, num_tokens - 1].detach()
                elif "mlp" in name:
                    num_tokens = list(output[0].size())[0]  # [num_tokens, 3072] for values;
                    self.model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                    self.model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "residual" in name or "embedding" in name:
                num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                if name == "layer_residual_" + str(self.num_layers-1):
                    self.model.activations_[name] = self.model.activations_[
                                                        "intermediate_residual_" + str(final_layer)] + \
                                                    self.model.activations_["mlp_" + str(final_layer)]

                else:
                    if 'out' in name:
                        self.model.activations_[name] = output[0][num_tokens-1].detach()
                    else:
                        self.model.activations_[name] = input[0][:,
                                                            num_tokens - 1].detach()

        return hook

    def reset_activations(self):
        self.model.activations_ = {}

        
class GPTJWrapper(ModelWrapper):

    def layer_decode(self, hidden_states):
        logits = []
        for i,h in enumerate(hidden_states):
            h=h[:, -1, :] #(batch, num tokens, embedding size) take the last token
            if i == len(hidden_states)-1:
                normed = h #ln_f would already have been applied
            else:
                normed = self.model.transformer.ln_f(h)
            l = torch.matmul(self.model.lm_head.weight, normed.T)
            logits.append(l)
        return logits

    def add_hooks(self):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook') 
            self.hooks.append(self.model.transformer.h[i].ln_1.register_forward_hook(self.get_activation(f'in_sln_{i}')))
            self.hooks.append(self.model.transformer.h[i].attn.register_forward_hook(self.get_activation('attn_'+str(i))))
            self.hooks.append(self.model.transformer.h[i].mlp.register_forward_hook(self.get_activation("intermediate_residual_" + str(i))))
            self.hooks.append(self.model.transformer.h[i].mlp.register_forward_hook(self.get_activation('mlp_'+str(i))))
            #print(self.model.activations_)


class GPT2Wrapper(ModelWrapper):

    def layer_decode(self, hidden_states):
        logits = []
        for i,h in enumerate(hidden_states):
            h=h[:, -1, :] #(batch, num tokens, embedding size) take the last token
            if i == len(hidden_states)-1:
                normed = h #ln_f would already have been applied
            else:
                normed = self.model.transformer.ln_f(h)
            l = torch.matmul(self.model.lm_head.weight, normed.T)
            logits.append(l)
        return logits

    def add_hooks(self):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook') 
            self.hooks.append(self.model.transformer.h[i].ln_1.register_forward_hook(self.get_activation(f'in_sln_{i}')))
            self.hooks.append(self.model.transformer.h[i].attn.register_forward_hook(self.get_activation('attn_'+str(i))))
            self.hooks.append(self.model.transformer.h[i].ln_2.register_forward_hook(self.get_activation("intermediate_residual_" + str(i))))
            self.hooks.append(self.model.transformer.h[i].ln_2.register_forward_hook(self.get_activation("out_intermediate_residual_" + str(i))))
            self.hooks.append(self.model.transformer.h[i].mlp.register_forward_hook(self.get_activation('mlp_'+str(i))))
            #print(self.model.activations_)


    def get_pre_wo_activation(self, name):
        #wo refers to the output matrix in attention layers. The last linear layer in the attention calculation

        def hook(module, input, output):
            #use_cache=True (default) and output_attentions=True have to have been passed to the forward for this to work
            _, past_key_value, attn_weights = output
            value = past_key_value[1]
            pre_wo_attn = torch.matmul(attn_weights, value)    
            self.model.activations_[name]=pre_wo_attn

        return hook

    def get_past_layer(self, name):
        #wo refers to the output matrix in attention layers. The last linear layer in the attention calculation

        def hook(module, input, output):
            #use_cache=True (default) and output_attentions=True have to have been passed to the forward for this to work
            #print(len(output), output, name)
            _, past_key_value, attn_weights = output  
            self.layer_pasts[name]=past_key_value

        return hook

    def add_mid_attn_hooks(self):
        for i in range(self.num_layers):
            self.hooks.append(self.model.transformer.h[i].attn.register_forward_hook(self.get_pre_wo_activation('mid_attn_'+str(i))))

            self.hooks.append(self.model.transformer.h[i].attn.register_forward_hook(self.get_past_layer('past_layer_'+str(i))))

    def rm_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def reset_activations():
        self.activations_ = {}
        self.last_pasts = {}
           


class BloomWrapper(ModelWrapper):
    def layer_decode(self, hidden_states):
        logits = []
        for i,h in enumerate(hidden_states):
            h=h[:, -1, :] #(batch, num tokens, embedding size) take the last token
            if i == len(hidden_states)-1:
                normed = h #ln_f would already have been applied
            else:
                normed = self.model.transformer.ln_f(h)
            
            l = torch.matmul(self.model.lm_head.weight, normed.T)
            logits.append(l)
        return logits

    def add_hooks(self):
        for i in range(self.num_layers):
            #intermediate residual between
            #print('saving hook')
            #self.hooks.append(self.model.transformer.h[i].ln_1.register_forward_hook(self.get_activation(f'in_sln_{i}')))
            self.hooks.append(self.model.transformer.h[i].self_attention.register_forward_hook(self.get_activation('attn_'+str(i))))
            self.hooks.append(self.model.transformer.h[i].mlp.register_forward_hook(self.get_activation("intermediate_residual_" + str(i))))
            self.hooks.append(self.model.transformer.h[i].mlp.register_forward_hook(self.get_activation('mlp_'+str(i))))

class BloomPetalsWrapper(BloomWrapper):
    def get_layers(self, tokens, **kwargs):
        outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)
        hidden_states, true_logits = outputs.hidden_states, outputs.logits #hidden states will be none unfortunately.
        logits = [true_logits.squeeze(0)[-1].unsqueeze(-1),] #no real reason for this weirdness
        return torch.stack(logits).squeeze(-1)#, true_logits.squeeze(0)

    #note: attention and mlp outputs have residual already added in bloom. Need to subtract input from output to get effect
    #see here: https://github.com/huggingface/transformers/blob/983e40ac3b2af68fd6c927dce09324d54d023e54/src/transformers/models/bloom/modeling_bloom.py#L212
