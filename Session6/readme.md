
## END 2 Phase 1 Assignment 6 - How to make encoder-decoder classes
------------------------------------------------------------------------------------------------------------

## Group : 
1. Sunny Sinha
2. Pratik Jain
3. Anudeep

----------------------
## Notes 
---------------------------------------------------------------------------------------------------------------------------

## Question
* Make 2 seperate classes for encoder, decoder and 1 class for combining logic
* Design the network so that it follows following logic flow:

  - embedding
  - word from a sentence +last hidden vector -> encoder -> single vector
  - single vector + last hidden vector -> decoder -> single vector
  - single vector -> FC layer -> Prediction

## Solution
* Created seperate classes for Encoder and Decoder and a 3rd class which combines the seperate outputs of encoder and decoder

#### Encoder
* Takes sentence text -> converts to embeddings -> runs through LSTM -> get final hidden vector(context vector representing full sentence information) as output
```python
class encoder_part(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
    super().__init__() 
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)

  def forward(self, text, text_lengths):
    embedded = self.embedding(text)
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
    packed_output, (hidden, cell) = self.encoder(packed_embedded)
    return packed_output, hidden
````

#### Decoder
* Defined as sequence of individual lstm cells which runs in a loop for specified no of times
* Takes encoder hidden vector as input
* This input vector is sent to sequence of lstm cells( same input vector for each cell along with last lstm cell hidden output)
* For first lstm cell hidden and cell states are initialized to 0
* Final lstm cell's hidden output vector is the output from Decoder
```python
class decoder_part(nn.Module):

  def __init__(self, input_to_decoder_size, decoder_hidden_size, no_times_decoder_cell_has_to_run):
    super().__init__()
    self.decoder_single_rnn_cell = nn.LSTMCell(input_to_decoder_size,decoder_hidden_size)
    self.no_times_decoder_cell_has_to_run = no_times_decoder_cell_has_to_run
    self.decoder_hidden_size = decoder_hidden_size

  def forward(self, encoder_context_vector):
    encoder_context_vector = encoder_context_vector.squeeze(0)
    hx = torch.zeros(encoder_context_vector.size(0),self.decoder_hidden_size).to(device)
    cx = torch.zeros(encoder_context_vector.size(0),self.decoder_hidden_size).to(device)
    otpt = []
    for i in range(self.no_times_decoder_cell_has_to_run):
      hx,cx = self.decoder_single_rnn_cell(encoder_context_vector,(hx,cx))
      otpt.append(hx)
    otpt = torch.stack(otpt,dim = 0)
    return otpt, hx
```

#### Combining encoder and decoder classes outputs
* Created a 3rd class to design the full logic and use encoder/decoder outputs
* In this class text is taken as input -> passed to encoder ->encoder output passed to decoder -> decoder output passed to fully connected layer to give output of size equals to no of classification classes.
```python
class combining_encoder_decoder(nn.Module):
  
  def __init__(self, encoder, decoder, hidden_dim, output_dim):
      super().__init__()
      self.encoder = encoder
      self.decoder = decoder
      self.fc = nn.Linear(hidden_dim, output_dim)
  
  def forward(self,src,src_len):
    enc_packed_outputs, enc_hidden = self.encoder(src,src_len)
    dec_otpt, dec_hidden = self.decoder(enc_hidden)
    dense_outputs = self.fc(dec_hidden)
    op = F.softmax(dense_outputs, dim=1)
    return op
 ```
 
 ## Training logs
```python
Train Loss: 1.080 | Train Acc: 55.53%
 Val. Loss: 1.038 |  Val. Acc: 69.20% 

Train Loss: 0.982 | Train Acc: 69.12%
 Val. Loss: 0.895 |  Val. Acc: 68.30% 

Train Loss: 0.873 | Train Acc: 69.12%
 Val. Loss: 0.850 |  Val. Acc: 68.30% 

Train Loss: 0.829 | Train Acc: 72.67%
 Val. Loss: 0.826 |  Val. Acc: 74.11% 

Train Loss: 0.794 | Train Acc: 77.70%
 Val. Loss: 0.813 |  Val. Acc: 75.89% 

Train Loss: 0.766 | Train Acc: 79.90%
 Val. Loss: 0.794 |  Val. Acc: 76.34% 

Train Loss: 0.739 | Train Acc: 82.35%
 Val. Loss: 0.788 |  Val. Acc: 75.89% 

Train Loss: 0.716 | Train Acc: 84.12%
 Val. Loss: 0.785 |  Val. Acc: 76.79% 

Train Loss: 0.702 | Train Acc: 85.47%
 Val. Loss: 0.773 |  Val. Acc: 77.68% 

Train Loss: 0.687 | Train Acc: 86.49%
 Val. Loss: 0.760 |  Val. Acc: 79.46%
```

