from keras import backend as K

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, Lambda, Reshape
from keras.layers.pooling import MaxPooling2D
from random import randint
from load import to_load, create_roidb
from load import batch_generator

##########################################################
# RoiDB setup
print("Initializing RoiDB...")
roidb = create_roidb(to_load)
print("RoiDB initialization complete!")


##########################################################
# MODEL SETUP

# Setup input shape assuming Tensorflow backend
print("Setting up models..")
images = Input([107,107,3], name='Images')

# Use VGG-16 As a base model
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=images)
# for l in base_model.layers:
#     l.trainable=False # Disable for these convolutional layers

# Add the shared fully connected layers
shared = Flatten()(base_model.output)
fc1 = Dense(512, activation='relu')(shared)
fc2 = Dense(512, activation='relu')(fc1)

D = len(roidb) # The number of domains
logits = [Dense(1, activation='sigmoid')(fc2) for _ in range(D)]
models = [Model(input=base_model.input, output=logit) for logit in logits]

# Set the optimizer and the loss
print("Compiling models")
[m.compile(optimizer='adam', loss='binary_crossentropy') for m in models]
print("Models are setup!")

##########################################################
# TRAINING

iters = 5 # The number of times each sequence should be trained on

print("Starting Training")
gen=batch_generator(roidb)
for _ in range(iters):
    for (i,m) in enumerate(models):
        print("Training on dataset {}".format(i+1))
        X,Y = gen.next()
        m.fit(X,Y, nb_epoch=1, verbose=0)
        # Will probably use fit_generator (https://keras.io/models/model/#methods) in the actual thing

print("Training complete!")
##########################################################        
