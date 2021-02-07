import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Merge
from keras.optimizers import Adam, RMSprop
from keras.layers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import keras
from keras.applications.inception_v3 import InceptionV3


embedding_size = 300
max_len = 40
vocab_size = 8256

image_model = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        RepeatVector(max_len)
    ])

caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(embedding_size))
    ])

final_model = Sequential([
        Merge([image_model, caption_model], mode='concat', concat_axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(vocab_size),
        Activation('softmax')
    ])

final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

final_model.summary()


#load weights
weightFile = "time_inceptionV3_7_loss_3.2604.h5"
weightFile = "time_inceptionV3_1.88.h5"
weightFile = "time_inceptionV3_7_loss_2.092.h5"

final_model.load_weights(weightFile)

unique = pickle.load(open('unique.p', 'rb'))
word2idx = {val:index for index, val in enumerate(unique)}
word2idx['<start>']

idx2word = {index:val for index, val in enumerate(unique)}
idx2word[5553]


#load feature extraction model
model = InceptionV3(weights='imagenet')

#initialize model
from keras.models import Model

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc


def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e =  encode(image)
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])