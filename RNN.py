class RNNEncoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, bidir, n_layers, dropout=0.5):
        super(RNNEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidir = bidir
        assert type(self.bidir)==bool
        self.dropout=dropout
        
        self.embedding = embedder # nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidir)
        self.init_h = nn.Parameter(torch.randn(self.n_layers*(1+self.bidir), 1, self.hidden_size), requires_grad=True)#learnable h0
        self.init_weights()
        
    def init_weights(self):
        """adopted from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5"""
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if len(w.shape)>1: 
                weight_init.orthogonal_(w.data)
            else:
                weight_init.normal_(w.data)
                
    
    def forward(self, inputs, input_lens=None, init_h=None): 
        # init_h: [n_layers*n_dir x batch_size x hid_size]
        if self.embedding is not None:
            inputs=self.embedding(inputs)  # input: [batch_sz x seq_len] -> [batch_sz x seq_len x emb_sz]
        
        batch_size, seq_len, emb_size=inputs.size()
        inputs=F.dropout(inputs, self.dropout, self.training)# dropout
        
        if input_lens is not None:# sort and pack sequence 
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)        
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)
        
        if init_h is None:
            init_h = self.init_h.expand(-1,batch_size,-1).contiguous()# use learnable initial states, expanding along batches
        #self.rnn.flatten_parameters() # time consuming!!
        hids, h_n = self.rnn(inputs, init_h) # hids: [b x seq x (n_dir*hid_sz)]  
                                                  # h_n: [(n_layers*n_dir) x batch_sz x hid_sz] (2=fw&bw)
        if input_lens is not None: # reorder and pad
            _, inv_indices = indices.sort()
            hids, lens = pad_packed_sequence(hids, batch_first=True)     
            hids = hids.index_select(0, inv_indices)
            h_n = h_n.index_select(1, inv_indices)
        h_n = h_n.view(self.n_layers, (1+self.bidir), batch_size, self.hidden_size) #[n_layers x n_dirs x batch_sz x hid_sz]
        h_n = h_n[-1] # get the last layer [n_dirs x batch_sz x hid_sz]
        enc = h_n.view(batch_size,-1) #[batch_sz x (n_dirs*hid_sz)]
            
        return enc, hids
    
class ContextEncoder(nn.Module):
    def __init__(self, utt_encoder, input_size, hidden_size, n_layers=1, dropout=0.5):
        super(ContextEncoder, self).__init__()     
        self.utt_encoder=utt_encoder
        self.ctx_encoder= RNNEncoder(None, input_size, hidden_size, False, n_layers, dropout)

    def forward(self, context, context_lens, utt_lens): # context: [batch_sz x diag_len x max_utt_len] 
                                                      # context_lens: [batch_sz x dia_len]
        batch_size, max_context_len, max_utt_len = context.size()
        utts = context.view(-1, max_utt_len) # [(batch_size*diag_len) x max_utt_len]
        utt_lens = utt_lens.view(-1)
        utt_encs, _ = self.utt_encoder(utts, utt_lens) # [(batch_size*diag_len) x 2hid_size]
        
        utt_encs = utt_encs.view(batch_size, max_context_len, -1)
        enc, hids = self.ctx_encoder(utt_encs, context_lens)
        return enc
  

class RNNDecoder(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, vocab_size, n_layers=1, dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.n_layers = n_layers
        self.input_size= input_size # size of the input to the RNN (e.g., embedding dim)
        self.hidden_size = hidden_size # RNN hidden size
        self.vocab_size = vocab_size # RNN output size (vocab size)
        self.dropout= dropout
        
        self.embedding = embedder
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.project = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        for w in self.rnn.parameters(): # initialize the gate weights with orthogonal
            if w.dim()>1:
                weight_init.orthogonal_(w)
        self.project.weight.data.uniform_(-0.1, 0.1)#nn.init.xavier_normal_(self.out.weight)        
        nn.init.constant_(self.project.bias, 0.)

    def forward(self, init_h, inputs=None, lens=None, enc_hids=None, src_pad_mask=None, context=None):
        '''
        init_h: initial hidden state for decoder
        enc_hids: enc_hids for attention use
        context: context information to be paired with input
        inputs: inputs to the decoder
        lens: input lengths
        '''
        if self.embedding is not None:
            inputs = self.embedding(inputs) # input: [batch_sz x seqlen x emb_sz]
        batch_size, maxlen, _ = inputs.size()
        inputs = F.dropout(inputs, self.dropout, self.training)  
        h = init_h.unsqueeze(0) # last_hidden of decoder [n_dir x batch_sz x hid_sz]        

        if context is not None:            
            repeated_context = context.unsqueeze(1).repeat(1, maxlen, 1) # [batch_sz x max_len x hid_sz]
            inputs = torch.cat([inputs, repeated_context], 2)
                
            #self.rnn.flatten_parameters()
        hids, h = self.rnn(inputs, h)         
        decoded = self.project(hids.contiguous().view(-1, self.hidden_size))# reshape before linear over vocab
        decoded = decoded.view(batch_size, maxlen, self.vocab_size)
        return decoded, h
    
    def sampling(self, init_h, enc_hids, src_pad_mask, context, maxlen, to_numpy=True):
        """
        A simple greedy sampling
        :param init_h: [batch_sz x hid_sz]
        :param enc_hids: a tuple of (enc_hids, mask) for attention use. [batch_sz x seq_len x hid_sz]
        """
        device = init_h.device
        batch_size = init_h.size(0)
        decoded_words = torch.zeros((batch_size, maxlen), dtype=torch.long, device=device)  
        sample_lens = torch.zeros((batch_size), dtype=torch.long, device=device)
        len_inc = torch.ones((batch_size), dtype=torch.long, device=device)
               
        x = torch.zeros((batch_size, 1), dtype=torch.long, device=device).fill_(SOS_ID)# [batch_sz x 1] (1=seq_len)
        h = init_h.unsqueeze(0) # [1 x batch_sz x hid_sz]  
        for di in range(maxlen):  
            if self.embedding is not None:
                x = self.embedding(x) # x: [batch_sz x 1 x emb_sz]
            h_n, h = self.rnn(x, h) # h_n: [batch_sz x 1 x hid_sz] h: [1 x batch_sz x hid_sz]

            logits = self.project(h_n) # out: [batch_sz x 1 x vocab_sz]  
            logits = logits.squeeze(1) # [batch_size x vocab_size]                  
            x = torch.multinomial(F.softmax(logits, dim=1), 1)  # [batch_size x 1 x 1]?
            decoded_words[:,di] = x.squeeze()
            len_inc=len_inc*(x.squeeze()!=EOS_ID).long() # stop increse length (set 0 bit) when EOS is met
            sample_lens=sample_lens+len_inc            
        
        if to_numpy:
            decoded_words = decoded_words.data.cpu().numpy()
            sample_lens = sample_lens.data.cpu().numpy()
        return decoded_words, sample_lens

class MyModel(nn.Module):
    '''The basic Hierarchical Recurrent Encoder-Decoder model. '''
    def __init__(self, config, vocab_size):
        super(MyModel, self).__init__()
        self.vocab_size = vocab_size 
        self.maxlen=config['maxlen']
        self.clip = config['clip']
        self.init_w = config['init_w']
        
        self.embedder= nn.Embedding(vocab_size, config['emb_size'], padding_idx=PAD_ID)
        self.utt_encoder = RNNEncoder(self.embedder, config['emb_size'], config['rnn_hid_utt'], True, 
                                   config['n_layers'], config['dropout']) 
                                                        # utter encoder: encode response to vector
        self.context_encoder = ContextEncoder(self.utt_encoder, config['rnn_hid_utt']*2,
                                              config['rnn_hid_ctx'], 1, config['dropout']) 
                                              # context encoder: encode context to vector    
        self.decoder = RNNDecoder(self.embedder, config['emb_size'], config['rnn_hid_ctx'], vocab_size, 1, config['dropout']) # utter decoder: P(x|c,z)
        self.optimizer = optim.Adam(list(self.context_encoder.parameters())
                                      +list(self.decoder.parameters()),lr=config['lr'])

    def forward(self, context, context_lens, utt_lens, response, res_lens):
        c = self.context_encoder(context, context_lens, utt_lens)
        output,_ = self.decoder(c, response[:,:-1], res_lens-1) # decode from z, c  # output: [batch x seq_len x n_tokens]   
        dec_target = response[:,1:].clone()
        dec_target[response[:,1:]==PAD_ID] = -100
        loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size), dec_target.view(-1))
        return loss
    
    def train_batch(self, context, context_lens, utt_lens, response, res_lens):
        self.context_encoder.train()
        self.decoder.train()
        
        loss = self.forward(context, context_lens, utt_lens, response, res_lens)
        
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` to prevent exploding gradient in RNNs
        nn.utils.clip_grad_norm_(list(self.context_encoder.parameters())+list(self.decoder.parameters()), self.clip)
        self.optimizer.step()
        
        return {'train_loss': loss.item()}      
    
    def valid(self, context, context_lens, utt_lens, response, res_lens):
        self.context_encoder.eval()  
        self.decoder.eval()        
        loss = self.forward(context, context_lens, utt_lens, response, res_lens)
        return {'valid_loss': loss.item()}
    
    def sample(self, context, context_lens, utt_lens, n_samples):    
        self.context_encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            c = self.context_encoder(context, context_lens, utt_lens)
        sample_words, sample_lens = self.decoder.sampling(c, None, None, None, n_samples, self.maxlen)  
        return sample_words, sample_lens  