import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import normalizePhoto
from config import *
from PIL import Image
import normalizeText
import dataset
epochs = 100
data_dir = pathlib.Path('./dataset')
data_text_dir = pathlib.Path('./text_dataset')
num_classes = len(os.listdir(data_dir))
name = 'gen_2'
max_features = 10000
sequence_length = 250


def learn(model_name):
    X, y = dataset.get_dataset()
    y = tf.convert_to_tensor(y)
    X = tf.convert_to_tensor(X)
    raw_train_ds = tf.data.Dataset.from_tensors((X,y))
    raw_train_ds.batch(batch_size)


    #raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        # data_text_dir,
        # batch_size=batch_size,
        # validation_split=0.2,
        # subset='training',
        # seed=123)
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        data_text_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=123)
    print(raw_val_ds)
    print(raw_train_ds)

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        data_text_dir,
        batch_size=batch_size)


    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_features + 1, 16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3)]
    )

    model.summary()
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )
    export_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        tf.keras.layers.Activation('sigmoid')
    ])
    export_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=['accuracy']    )
    export_model.save(f'models/{model_name}/')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def predict(model_name, image_path):
    model = tf.keras.models.load_model(f'models/{model_name}')
    img = normalizePhoto.normalizePhoto(image_path)
    text = normalizeText.get_text(img)
    text = normalizeText.normalize(text)
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)
    t = vectorize_layer.adapt([text])
    print(t,text)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    #input_arr = tf.keras.preprocessing.image.img_to_array(image)

    # input_arr = normalization_layer(input_arr)

    predictions = probability_model.predict([text])
    return predictions[0]

def test(model_name):
    data = '220120100022020110000010201020001210010101002100021100000000120110002010000001020020201020020210102000100' \
           '200020122100200010002020010201021110000010011010000000000101000102002000100000021001002020221120120021102' \
           '210100001110010200011000000110100001001001202100010001020110100020211120001000100001101100202020000012211' \
           '10'
    result = 0
    try:
        for i, img in enumerate(os.listdir('./test_images')):
            pred = predict(model_name, f'./test_images/{img}')
            print(img, pred, sep='\t', end='\t')
            Image.open( f'./test_images/{img}').show()
            input()
            # if pred[0] == max(pred) and data[i] == '0':
            #     result += 1
            #     print('Bill')
            # elif pred[1] == max(pred) and data[i] == '1':
            #     result += 1
            #     print('Facture')
            # elif pred[2] == max(pred) and data[i] == '2':
            #     result += 1
            #     print('Error')
            # else:
            #     print('X')
    except KeyboardInterrupt:
        print('Тестирование прервано')
    print(f'{round((result / len(data) * 100), 4)}% Accuracy')
learn("abobus")
test("abobus")