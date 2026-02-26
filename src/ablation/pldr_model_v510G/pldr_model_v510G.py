'''
Model Implementation for Large Language Model from Power Law Decoder Representations v510G (PLDR-LLM-v510G) with KV-cache and G-cache.
v510G removes the deep residual layers and custom weight/biases that learn A, A_LM and G_LM
G_LM is provided as a predefined value during model initialization.
'''

import torch
from torch import nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings

import power_law_attention_layer_v510G as plgatt


class plgMultiHeadAttention(nn.Module):
    '''
    Power Law Multihead Attention Implementation for PLDR-LLM.
    '''
    def __init__(self, d_model, num_heads, Gcache, max_seq_len=4096,  
                 activation=F.silu, device=None, **kwargs):

        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.activation=activation
        self.max_seq_len=max_seq_len
        self.device=device
        self.Gcache=Gcache

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model, bias=True, device=self.device)
        self.wq.apply(weights_init)
        self.wk = nn.Linear(d_model, d_model, bias=True, device=self.device)
        self.wk.apply(weights_init)
        self.wv = nn.Linear(d_model, d_model, bias=True, device=self.device)
        self.wv.apply(weights_init)

        self.plgatt_layer= plgatt.plga_layer(F_hidden=self.depth,
                                             Gcache=self.Gcache,
                                             device=self.device)

        self.dense = nn.Linear(d_model, d_model, bias=True, device=self.device)
        self.dense.apply(weights_init)

        #torchtune based rotary embedding expects of shape [batch, seqlen, num_heads, depth]
        self.rotary_embedding=RotaryPositionalEmbeddings(dim=self.depth, max_seq_len=self.max_seq_len, base=10000).to(device=self.device)

    def split_heads(self, x, batch_size):
        '''
        Split the last dimension into (num_heads, depth).
        '''
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x #(batch_size, seq_len, num_heads, depth)

    def forward(self, inputs, kvcache=None, **kwargs):
        '''
        Args:
            inputs: [q, k, v, mask]
            kvcache: [k, v]
        Returns:
            inductive and deductive task outputs.
        '''
        q, k, v, mask = inputs
        batch_size = q.size()[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)


        q = self.split_heads(q, batch_size)  # (batch_size, seq_len, num_heads, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)


        if kvcache is None:
            q = self.rotary_embedding(q)
            k = self.rotary_embedding(k)
        else:
            q = self.rotary_embedding(q, input_pos=kvcache[0].size()[1]) # (batch_size, seq_len, num_heads, depth) 
            k = self.rotary_embedding(k, input_pos=kvcache[0].size()[1]) 
            k=torch.concat([kvcache[0], k], axis=1) # (batch_size, seq_len+1, num_heads, depth)
            v=torch.concat([kvcache[1], v], axis=1)
        
        kvcache=[k, v]
        
        q = torch.permute(q, [0, 2, 1, 3]) #(batch_size, num_heads, seq_len, depth)
        k = torch.permute(k, [0, 2, 1, 3]) #(batch_size, num_heads, seq_len, depth)
        v = torch.permute(v, [0, 2, 1, 3]) #(batch_size, num_heads, seq_len, depth)

        #Apply multi-head power law attention
        Hnext = self.plgatt_layer([q, k, v, mask])

        Hnext = torch.permute(Hnext, [0, 2, 1, 3])

        Hnext= Hnext.reshape(batch_size, -1, self.d_model) # [batch_size, seq_len, d_model]

        output = self.dense(Hnext)

        return output, kvcache


class PLDR_DecoderLayer(nn.Module):
    '''
    Single decoder layer implementation for PLDR-LLM with single masked multihead attention.
    '''
    def __init__(self, d_model, num_heads, dff, Gcache, max_seq_len=4096, 
                 activation=F.silu, device=None, **kwargs):

        super().__init__(**kwargs)

        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.activation=activation
        self.max_seq_len=max_seq_len
        self.device=device
        self.Gcache=Gcache

        self.mha1 = plgMultiHeadAttention(d_model=self.d_model, 
                                          num_heads=self.num_heads,
                                          Gcache=self.Gcache,
                                          max_seq_len=self.max_seq_len,  
                                          activation=self.activation,
                                          device=self.device)

        self.ffn = self.dec_point_wise_feed_forward_network()

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=1e-6, device=self.device)
        self.layernorm2 = nn.LayerNorm(self.d_model, eps=1e-6,  device=self.device)

    def forward(self, inputs, kvcache=None, **kwargs):
        '''
        inputs: [x, look_ahead_mask]
        kvcache: [k, v]
        Returns Decoder Layer output and deductive task outputs.
        '''

        x, look_ahead_mask = inputs

        attn1, kvcache = self.mha1([x,x,x, look_ahead_mask], kvcache=kvcache)
        out1 = self.layernorm1(attn1 + x)

        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

        return out2, kvcache


    #GLUVariant implementation for feedforward network, scale dff accordingly (i.e., 2/3 of original).
    def dec_point_wise_feed_forward_network(self):
        return GLUVariant(self.d_model, self.dff, self.d_model, activation=self.activation, device=self.device)


class PLDR_Decoder(nn.Module):
    '''
    Multi layer decoder implementation for PLDR-LLM
    '''
    def __init__(self, num_layers, d_model, num_heads, Gcachelst, dff, target_vocab_size,
                  max_seq_len=4096, activation=F.silu, device=None, **kwargs):

        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads=num_heads
        self.dff=dff
        self.target_vocab_size = target_vocab_size
        self.activation=activation
        self.max_seq_len=max_seq_len
        self.device=device
        self.Gcachelst=Gcachelst

        self.embedding = nn.Embedding(self.target_vocab_size, self.d_model, device=self.device)

        self.dec_layers = nn.ModuleList([PLDR_DecoderLayer(d_model=self.d_model, 
                                             num_heads=self.num_heads, 
                                             dff=self.dff,
                                             Gcache=self.Gcachelst[i].to(self.device),
                                             max_seq_len=self.max_seq_len,  
                                             activation=self.activation,
                                             device=self.device) for i in range(self.num_layers)])

        self.layernorm1 = nn.LayerNorm(self.d_model, eps=1e-6, device=self.device)


    def forward(self, inputs, kvcachelst=None, **kwargs):
        '''
        inputs: [x, look_ahead_mask].
        kvcachelst: list of KV-caches.
        Returns:
        Output of decoder and list of KV-caches for PLDR_Decoder.
        '''
 
        x, look_ahead_mask= inputs

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= torch.sqrt(torch.tensor(self.d_model).to(torch.float32))

        x=self.layernorm1(x)

        upd_kvcachelst=[]
        for i in range(self.num_layers):
            x, upd_kvcache= self.dec_layers[i]([x, look_ahead_mask], kvcache=kvcachelst[i])
            upd_kvcachelst.append(upd_kvcache)

        return x, upd_kvcachelst


class PLDR_Model(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, Gcachelst, input_vocab_size,
                  max_seq_len=4096, activation=F.silu, device=None, **kwargs):
        '''
        Args:
            num_layers: Number of Decoder Layers.
            d_model: Embedding/LM feature dimension
            num_heads: Number of power law attention heads
            dff: Number of neurons on single layer of fully connected network
            input_vocab_size: Vocabulary size for the embedding layer (target_vocab_size for PLDR_Decoder class)
            max_seq_len: maximum sequence length for Rotary Embedding
            activation: Activation for FFN and GLU Variant layer.
            device: Device to load tensors on.
        Returns:
            Logit probabilities for predicted sentence, decoder output and
            list of KV-caches.
        '''
        super().__init__(**kwargs)

        self.num_layers = num_layers
        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.input_vocab_size = input_vocab_size
        self.activation=activation
        self.max_seq_len=max_seq_len
        self.device=device
        self.Gcachelst=Gcachelst

        self.decoder = PLDR_Decoder(num_layers=self.num_layers, 
                                    d_model=self.d_model, 
                                    num_heads=self.num_heads,
                                    Gcachelst=self.Gcachelst,
                                    dff=self.dff,
                                    target_vocab_size=self.input_vocab_size,
                                    max_seq_len=self.max_seq_len, 
                                    activation=self.activation,
                                    device=self.device)

        self.final_layer = nn.Linear(self.d_model, self.input_vocab_size, bias=True, device=self.device)
        self.final_layer.apply(weights_init)

    def forward(self, inputs, kvcachelst=None, **kwargs):
        '''
        inputs: [inp, look_ahead_mask]
        kvcachelst: list of KV-caches.
        Returns:
        Logit probabilites, decoder output, list of KV-caches        
        '''

        inp, look_ahead_mask=inputs

        kvcachelst = kvcachelst if kvcachelst is not None else [None]*self.num_layers

        dec_output, upd_kvcachelst = self.decoder([inp, look_ahead_mask], kvcachelst=kvcachelst, Gcachelst=self.Gcachelst )

        final_output = self.final_layer(dec_output)
        
        return final_output, dec_output, upd_kvcachelst

class GLUVariant(nn.Module):
    '''
    Implementation of GLU variants with default activation for SwiGLU configuration 
    For the hidden layer dff, to match size with non-SwiGLU FFN version scaling with 2/3 may be useful.
    '''
    def __init__(self, d_model, dff, depth,
                 activation=F.silu, device=None, **kwargs):
        super().__init__(**kwargs)
        self.dff=dff
        self.depth=depth
        self.d_model=d_model
        self.activation=activation
        self.device=device

        self.gluw1=nn.Linear(self.d_model, self.dff, bias=True, device=self.device)
        self.gluw1.apply(weights_init)
        self.gluw2=nn.Linear(self.d_model, self.dff, bias=True, device=self.device)
        self.gluw2.apply(weights_init)
        self.gluw3=nn.Linear(self.dff, self.depth, bias=True, device=self.device)
        self.gluw3.apply(weights_init)

    def forward(self, input, **kwargs):
        x1=self.gluw1(input)
        x1=self.activation(x1)
        x2=self.gluw2(input)
        return self.gluw3(torch.mul(x1, x2))

def weights_init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight.data)
        nn.init.zeros_(layer.bias.data)


