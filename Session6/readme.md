
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

### Encoder
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


