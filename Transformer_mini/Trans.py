import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

def load_data(path, num_examples=None):
    with open(path, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    if num_examples:
        lines = lines[:num_examples]
    eng, fra = zip(*[line.split('\t')[:2] for line in lines if len(line.split('\t')) >= 2])

    fra = [f"<start> {line} <end>" for line in fra]
    return list(eng), list(fra)

def tokenize(lang, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, filters='', oov_token='<OOV>')
    tokenizer.fit_on_texts(lang)
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = pad_sequences(tensor, padding='post')
    return tensor, tokenizer

# Load and tokenize
eng, fra = load_data('D:\\Downloads\\fra-eng\\fra.txt', 100000)
eng_tensor, eng_tok = tokenize(eng)
fra_tensor, fra_tok = tokenize(fra)

BUFFER_SIZE = len(eng_tensor)
BATCH_SIZE = 64
steps_per_epoch = BUFFER_SIZE // BATCH_SIZE

dataset = tf.data.Dataset.from_tensor_slices((eng_tensor, fra_tensor))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Define model
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state

class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, enc_output, hidden):
        x = self.embedding(x)
        x, state = self.gru(x, initial_state=hidden)
        x = self.fc(x)
        return x, state

class Seq2Seq(tf.keras.Model):
    def __init__(self, enc_vocab, dec_vocab, embedding_dim, units):
        super().__init__()
        self.encoder = Encoder(enc_vocab, embedding_dim, units)
        self.decoder = Decoder(dec_vocab, embedding_dim, units)

    def call(self, inp, tar, training=False):
        enc_output, enc_state = self.encoder(inp)
        dec_input = tar[:, :-1]
        predictions, _ = self.decoder(dec_input, enc_output, enc_state)
        return predictions

# Create model
embedding_dim = 256
units = 512
model = Seq2Seq(len(eng_tok.word_index) + 1, len(fra_tok.word_index) + 1, embedding_dim, units)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Training step
@tf.function
def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions = model(inp, tar, training=True)
        loss = loss_object(tar[:, 1:], predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Train
for epoch in range(50):  # Increased to 30 epochs
    total_loss = 0
    for (batch, (inp, tar)) in enumerate(dataset):
        batch_loss = train_step(inp, tar)
        total_loss += batch_loss
    print(f"Epoch {epoch+1}, Loss: {total_loss / steps_per_epoch:.4f}")

def translate(sentence):
    sentence = preprocess_sentence(sentence)
    

    tokens = eng_tok.texts_to_sequences([sentence])
   

    tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=eng_tensor.shape[1], padding='post')

    # Encode the input sentence
    enc_output, enc_state = model.encoder(tokens)

    # Start token as initial decoder input
    dec_input = np.array([[fra_tok.word_index['<start>']]])
    dec_hidden = enc_state
    output_sentence = []

    for i in range(20):
        # Run one step of decoder
        predictions, dec_hidden = model.decoder(dec_input, enc_output, dec_hidden)

        # Get predicted word ID
        next_id = tf.argmax(predictions[0, -1, :]).numpy()
        next_word = fra_tok.index_word.get(next_id, '')

       
        if next_word == '<end>':
            break

        if next_word not in ('<start>', ''):
            output_sentence.append(next_word)

        # Update decoder input for next step
        dec_input = np.array([[next_id]])

    return ' '.join(output_sentence)


# Take user input
while True:
    input_text = input("\nEnter English sentence to translate (or type 'exit'): ")
    if input_text.lower() == 'exit':
        break
    translated = translate(input_text)
    print("French:", translated)
M