import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import subprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import r
e, string, nltk
import t
ensorflow as tf
from nlt
k.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.callbacks import EarlyStopping

# Load datasets
df2 = pd.read_csv('/content/Twitter_Data.csv')
df1 = pd.read_csv('/content/Reddit_Data.csv')

df2.columns = ['messages', 'labels']
df1.columns = ['messages', 'labels']

# Merge datasets
df = pd.concat([df1, df2], ignore_index=True)

# Add length column
df['length'] = df['messages'].str.len()

# Plot histogram
sns.set_style('whitegrid')
plt.hist(df['length'], bins=100)
plt.show()

# Convert labels
df['labels'] = df['labels'].map({-1: 'negative', 0: 'neutral', 1: 'positive'})

# Drop NaN values
df = df.dropna()

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')

# Text preprocessing functions
def clean_text(text):
    text = str(text).lower()
    text = re.sub('https?:\/\/\S*|www\.\S+', 'URL', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('@\S*', 'user', text, flags=re.IGNORECASE)
    text = re.sub('^[+-]*?\d{1,3}[- ]*?\d{1,10}|\d{10}', 'NUMBER', text)
    text = re.sub('<3', 'HEART', text)
    text = re.sub('\w*\d+\w*', '', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
    text = ''.join([word for word in text if word not in string.punctuation])
    lm = WordNetLemmatizer()
    text = ' '.join([lm.lemmatize(word, pos='v') for word in text.split()])
    return text

# Apply text cleaning
df['messages_clean'] = df['messages'].apply(clean_text)

# Prepare data for training
X = df['messages_clean'].values
y = pd.get_dummies(df['labels']).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Tokenization
max_vocab_size = 50000
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
sequence_Xtrain = tokenizer.texts_to_sequences(X_train)
sequence_Xtest = tokenizer.texts_to_sequences(X_test)

V = len(tokenizer.word_index)
data_train = pad_sequences(sequence_Xtrain)
T = data_train.shape[1]

# Pad test sequences
data_test = pad_sequences(sequence_Xtest, maxlen=T)

# Model architecture
D = 20
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

i = Input(shape=(T,))
x = Embedding(V + 1, D)(i)
x = LSTM(128, dropout=0.2)(x)  # Fixed dropout argument
x = Dense(3, activation='softmax')(x)

model = Model(i, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(data_train, y_train, batch_size=128, validation_data=(data_test, y_test), epochs=5, callbacks=[early_stop])

# Model summary and evaluation
model.summary()
model.evaluate(data_test, y_test)

# Plot results
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.plot(history.history['accuracy'], label='accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# Make predictions
predictions = model.predict(data_test).argmax(axis=1)
print(classification_report(y_test.argmax(axis=1), predictions))

# Save model
model.save('model#1.h5')
