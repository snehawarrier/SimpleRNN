from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
max_features = 5000  # Only use top 5000 words
maxlen = 500         # Pad all sequences to length 500

(x_train, y_train), (_, _) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=32, input_length=maxlen))
model.add(SimpleRNN(units=32))  # No time_major used
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model briefly (just enough to get a working model)
model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.2)

# Save the new model without time_major
model.save("simple_rnn_imdb_fixed.h5")
print("✅ Model saved as simple_rnn_imdb_fixed.h5")
