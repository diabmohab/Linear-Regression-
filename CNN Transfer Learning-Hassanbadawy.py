#----------------------------------------------
#%%             Import Libraries
#----------------------------------------------

import numpy as np
import keras as k

#----------------------------------------------
#%%       Load and Test ResNet50
#----------------------------------------------
RESNET50_WEIGHTS = 'keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
RESNET50_NOTOP_WEIGHTS = 'keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Load Keras' ResNet50 model that was pre-trained against the ImageNet database
model = k.applications.resnet50.ResNet50(weights=RESNET50_WEIGHTS)

# Load the image file, resizing it to 224x224 pixels (required by this model)
img = k.preprocessing.image.load_img("bay.jpg", target_size=(224, 224))

# Convert the image to a numpy array
x = k.preprocessing.image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

#----------------------------------------------
#%%               Processing
#----------------------------------------------

# Scale the input image to the range used in the trained network
x = k.applications.resnet50.preprocess_input(x)

#----------------------------------------------
#%%               Prediction
#----------------------------------------------
# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = k.applications.resnet50.decode_predictions(predictions, top=9)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood))

#----------------------------------------------
#%%           Transfer Learning
#----------------------------------------------
num_class = 2
pretrained_model = k.applications.resnet50.ResNet50(weights=RESNET50_WEIGHTS)
print('Output_layer_type= {}'.format(pretrained_model.layers[-1]))
print('Output_layer_shape= {}'.format(pretrained_model.layers[-1].output_shape))
pretrained_model.layers.pop()
print('Output_layer_type= {}'.format(pretrained_model.layers[-1]))
print('Output_layer_shape= {}'.format(pretrained_model.layers[-1].output_shape))
for layer in pretrained_model.layers:
    layer.trainable = False
model = k.models.Sequential()
model.add(pretrained_model)
# model.add(k.layers.Flatten())
# model.add(k.layers.GlobalAveragePooling2D())
# model.add(k.layers.Dense(1024, activation='relu'))
model.add(k.layers.Dense(num_class, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
callbacks_list = [k.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
model.summary()

print('Input Shape = {}'.format(model.layers[0].input_shape))
print('Shape Shape = {}'.format(model.layers[-1].output_shape))
#----------------------------------------------
#%%           IMAGE PREPROCESSING
#----------------------------------------------
from keras.preprocessing.image import ImageDataGenerator

training_set = 'G:\\ML\ML\\Udimy\\ML\\Convolutional_Neural_Networks\\dataset\\training_set'
test_set = 'G:\\ML\ML\\Udimy\\ML\\Convolutional_Neural_Networks\\dataset\\test_set'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(training_set,
                                                 target_size = (224, 224),
                                                 batch_size = 32)

test_set = test_datagen.flow_from_directory(test_set,
                                            target_size = (224, 224),
                                            batch_size = 32)
#----------------------------------------------
#%%           IMAGE PREPROCESSING
#----------------------------------------------
model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)