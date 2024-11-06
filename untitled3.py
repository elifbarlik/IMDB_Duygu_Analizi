import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from keras.datasets import imdb #Keras'tan indirilebilen hazır IMDb veri seti. 
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from keras.preprocessing.sequence import pad_sequences


(x_train, y_train) , (x_test, y_test) = imdb.load_data(path = 'imdb.npz')

print("x_train Type: ", type(x_train))
print("y_train Type: ", type(y_train))

print('x_train Shape: ', x_train.shape)
print('y_train Shape: ', y_train.shape)

#%% EDA

print('y train values: ',np.unique(y_train))
print('y test values: ',np.unique(y_test))

uniqe_tr, counts_tr = np.unique(y_train, return_counts=True)
print('y train distribution: ', dict(zip(uniqe_tr, counts_tr)))

uniqe_te, counts_te = np.unique(y_test, return_counts=True)
print('y test distribution: ', dict(zip(uniqe_te, counts_te)))

plt.figure()
sns.barplot(x=uniqe_tr, y=counts_tr)
plt.xlabel('Classes')
plt.ylabel('Freq')
plt.title('Y train')

plt.figure()
sns.barplot(x=uniqe_te, y=counts_te)
plt.xlabel('Classes')
plt.ylabel('Freq')
plt.title('Y test')


d = x_train[0]
print(d)
print(len(d))

review_len_train = []
review_len_test = []
for i, ii in zip(x_train, x_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))
    
plt.figure(figsize=(10, 6))
sns.histplot(review_len_train, bins=50, kde=True, color='blue', alpha=0.3, label='Train')
sns.histplot(review_len_test, bins=50, kde=True, color='red', alpha=0.3, label='Test')

plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.title('Review Length Distribution - Train vs Test')
plt.legend()
plt.grid(True)
plt.show()

print('Train mean: ',np.mean(review_len_train))
print('Train median: ',np.median(review_len_train))
print('Train mode: ', stats.mode(review_len_train))

word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))

for keys, values in word_index.items():
    if values == 22:
        print(keys)

def whatItSay(index=24):
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i-3, '!') for i in x_train[index]])
    #Sayılar i-3 ile ayarlanıyor. IMDb veri setinde ilk 3 sayı (0, 1, 2) özel amaçlar için ayrıldığından, bu sayılar kelimelere karşılık gelmez. Bu nedenle i-3 yapılır.
    #Eğer reverse_index içinde bir sayı bulunamazsa, o sayıya karşılık olarak '!' karakteri eklenir.
    print(decode_review)
    print(y_train[index])
    return decode_review

decoded_review = whatItSay(36)

#%% PREPROCESS
num_words = 20000
(x_train, y_train) , (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train, x_val ,y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

maxlen = 500

x_train = pad_sequences(x_train, maxlen=maxlen)
x_val = pad_sequences(x_val, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


#%% RNN

rnn = tf.keras.Sequential() #katmanları sırayla ekleyebileceğimiz bir model yaratırız
rnn.add(tf.keras.layers.Input(shape=(maxlen,)))
rnn.add(tf.keras.layers.Embedding(input_dim=num_words,output_dim = 32)) #her kelimeyi belirli bir boyutla (32) ifade eder
#Boyut sayısı ne kadar yüksekse, kelimenin anlamını ifade etmek için modelin o kadar çok bilgisi olur. Ancak bu boyutu çok büyük tutmak modelin karmaşıklığını artırır, eğitimi zorlaştırır.
rnn.add(tf.keras.layers.LSTM(64, activation='tanh',return_sequences=True))
#LSTM, geleneksel RNN'lerin karşılaştığı "vanishing gradient" (kaybolan gradyan) sorununu aşmak için geliştirilmiştir ve bu sayede uzun vadeli bağıntıları öğrenmede daha etkili olur.
#LSTM, klasik RNN'lerin aksine, zamanla kaybolan bilgileri hatırlamakta zorlanmaz.
#"hafıza hücreleri" adı verilen özel bir yapıya sahiptir. Bu hücreler, geçmiş verilerden gelen bilgiyi daha uzun süre saklayabilirler.
#LSTM'nin temel bileşeni olan "kapılar" (gates), hücreye ne zaman bilgi ekleyeceğini, ne zaman çıkaracağını ve ne zaman unutacağını belirler. 
rnn.add(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True))
rnn.add(tf.keras.layers.Dropout(0.5))
rnn.add(tf.keras.layers.LSTM(64, activation='tanh'))
rnn.add(tf.keras.layers.Dense(1, activation='sigmoid')) #Çıkış katmanı , Tek bir nöron olduğu için, ikili (binary) sınıflandırma yapılır.

print(rnn.summary())
rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = np.array(x_train)
y_train = np.array(y_train)
history = rnn.fit(x_train, y_train, validation_data = (x_val, y_val), epochs=5, batch_size = 32, verbose=1)

score = rnn.evaluate(x_test, y_test)
print('Accuracy: %',score[1]*100)

plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title("Acc")
plt.ylabel('Acc')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title("Loss")
plt.ylabel('Acc')
plt.xlabel('Epochs')
plt.legend()
plt.show()

