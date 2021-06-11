
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
* Make 2 seperate classes for encoder, decoder and 1 class for combining logic.
* Design the network so that it follows following logic flow:

  - embedding
  - word from a sentence +last hidden vector -> encoder -> single vector
  - single vector + last hidden vector -> decoder -> single vector
  - single vector -> FC layer -> Prediction

## Solution
* Created seperate classes for Encoder and Decoder and a 3rd class which combines the seperate outputs of encoder and decoder.

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
* Defined as sequence of individual lstm cells which runs in a loop for specified no of times.
* Takes encoder hidden vector as input
* This input vector is sent to sequence of lstm cells( same input vector for each cell along with last lstm cell hidden output.
* For first lstm cell hidden and cell states are initialized to 0
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


