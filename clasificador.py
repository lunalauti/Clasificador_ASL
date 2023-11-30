# Red Neuronal Convolucional CNN
# Clasificador de lenguaje de señas a lenguaje escrito
# https://www.kaggle.com/datasets/grassknoted/asl-alphabet/

import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam

from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D

from keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle

train_folder = 'ASL2/asl_alphabet_train'
all_data = []
for folder in os.listdir(train_folder):
    label_folder = os.path.join(train_folder, folder)

    if os.path.isdir(label_folder):
        onlyfiles = [{'label': folder, 'path': os.path.join(label_folder, f)} for f in os.listdir(label_folder) if
                     os.path.isfile(os.path.join(label_folder, f))]

        all_data += onlyfiles

train_folder = 'archive/asl_alphabet_train/asl_alphabet_train'
for folder in os.listdir(train_folder):
    label_folder = os.path.join(train_folder, folder)

    if os.path.isdir(label_folder):
        onlyfiles = [{'label': folder, 'path': os.path.join(label_folder, f)} for f in os.listdir(label_folder) if
                     os.path.isfile(os.path.join(label_folder, f))]
        all_data += onlyfiles

# Crea un DataFrame con la lista de datos recopilada
data_df = pd.DataFrame(all_data)
print(data_df)

# Divido en train, test y holdout
x_train,x_holdout = train_test_split(data_df, test_size= 0.30, random_state=42,stratify=data_df[['label']])
x_train,x_test = train_test_split(x_train, test_size= 0.25, random_state=42,stratify=x_train[['label']])

# Para ajustar el tamaño a las imagenes
img_width, img_height = 64, 64
batch_size = 256 # Tamaño de los lotes de imagenes
y_col = 'label'
x_col = 'path'
no_of_classes = len(data_df[y_col].unique())

# Establezco tres generadores de datos de imágenes para entrenamiento, validación y holdout
train_datagen = ImageDataGenerator(rescale = 1/255.0) # Normaliza de las imágenes dividiendo cada píxel por 255.0.
train_generator = train_datagen.flow_from_dataframe(
    dataframe=x_train,x_col=x_col, y_col=y_col,
    target_size=(img_width, img_height),class_mode='categorical', batch_size=batch_size,
    shuffle=False,
)

validation_datagen = ImageDataGenerator(rescale = 1/255.0)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=x_test, x_col=x_col, y_col=y_col,
    target_size=(img_width, img_height), class_mode='categorical', batch_size=batch_size,
    shuffle=False
)

holdout_datagen = ImageDataGenerator(rescale = 1/255.0)
holdout_generator = holdout_datagen.flow_from_dataframe(
    dataframe=x_holdout, x_col=x_col, y_col=y_col,
    target_size=(img_width, img_height), class_mode='categorical', batch_size=batch_size,
    shuffle=False
)

# Defino el modelo de red neuronal convolucional
model = Sequential()

model.add(Conv2D(32, (5,5),padding = 'Same',activation ='relu', input_shape = (64,64,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(29, activation = "softmax"))

model.compile(optimizer = Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss',patience=5)

batch_size=128

history = model.fit(train_generator,
                    epochs=10,
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks = [early_stop])

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

fig = plt.figure(figsize=(14,7))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.show()


fig = plt.figure(figsize=(14,7))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')

# Realiza predicciones en el holdout
predictions = model.predict(holdout_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=-1)
true_classes = holdout_generator.classes
class_labels = list(holdout_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Imprimo algunas predicciones y etiquetas verdaderas
for i in range(10):  
    print(f"Sample {i + 1} - True Label: {class_labels[true_classes[i]]}, Predicted Label: {class_labels[predicted_classes[i]]}")

# ---------------------------- VISUALIZACION ----------------------------------------

# Obtengo un solo lote de imágenes y etiquetas
image_batch, true_label_batch = holdout_generator.next()

num_images_to_visualize = 15 

plt.figure(figsize=(6, 15))
for i in range(num_images_to_visualize):

    image = image_batch[i]
    true_label = np.argmax(true_label_batch[i])  # Etiquetas son "one-hot encoding"

    # Realizo predicciones en la imagen actual
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_class = np.argmax(prediction)

    # Visualizo
    plt.subplot(4, 5, i + 1)
    plt.imshow(image)

    # Asigna las etiquetas verdaderas y predichas
    true_label = class_labels[true_label]
    predicted_label = class_labels[predicted_class]

    plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
    plt.axis('off')

plt.show()

# -------------------------------------------------------------------------------------

model.save('model.h5')

f = open('model2.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
