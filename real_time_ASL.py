import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np

class_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

pickleFile = open("model2.p", 'rb')
model2 = pickle.load(pickleFile)['model']

# Capturar una imagen
vid = cv2.VideoCapture(0)


old_predicted_class = -1

font = cv2.FONT_HERSHEY_SIMPLEX
position = (10,50)
fontScale = 2
fontColor = (255, 255, 0)
texto = ""

while(True):
    __, frame = vid.read()
    frame = cv2.flip(frame,1)

    # Display the resulting frame
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
  
    image = cv2.resize(image,(64,64))


    # plt.imshow(image)

# Realizar predicciones en la imagen actual
    prediction = model2.predict(np.expand_dims(image[:, :, :3], axis=0),verbose=0)
    predicted_class = np.argmax(prediction)
    
   
    if(old_predicted_class != predicted_class):
        old_predicted_class = predicted_class
        texto = class_labels[predicted_class]
        # print(class_labels[predicted_class])
    cv2.putText(frame, texto, position, font, fontScale, fontColor,3)
    cv2.imshow(f'predicción', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


# # # Visualizar la imagen
# image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
# image = cv2.resize(image,(64,64))
# plt.imshow(image)

# # Realizar predicciones en la imagen actual
# prediction = model2.predict(np.expand_dims(image[:, :, :3], axis=0))
# predicted_class = np.argmax(prediction)

# # Asignar las etiquetas verdaderas y predicha
# plt.title(f"Predicted: {class_labels[predicted_class]}")
# plt.axis('off')
# plt.show()