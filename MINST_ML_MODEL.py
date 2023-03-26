#!/usr/bin/env python
# coding: utf-8

# # ML model for the classification of handwritten digits using MNIST Dataset
# # Submitted by 
#     

# In[ ]:


import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
import matplotlib.pyplot as plt


# Splitting MNIST Datase

# In[ ]:


# Load datasets for training and testing.

(X_TRAIN,Y_TRAIN),(X_TEST,Y_TEST) = keras.datasets.mnist.load_data()


# In[ ]:


X_TEST.shape


# In[ ]:


X_TEST[44]


# In[1]:


Y_TRAIN[44]


# **Plotting some digits to check datasets**

# In[ ]:


import matplotlib.pyplot as plt
plt.imshow(X_TRAIN[44])


# In[ ]:


X_TRAIN = X_TRAIN/255
X_TEST = X_TEST/255


# In[ ]:


X_TRAIN[44]


# **Softmax layer basically gives probability of 10 output from softmax layer. After that we used argmax which gives the array index which have the maximum probability, and that is also the prediction**

# In[ ]:


model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))


# # Creating a simple neural network with 2 hidden layers with neurons, where I used Adam as optimizer and CrossEntropyLoss as a loss function.

# In[ ]:


model.summary()


# In[ ]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# **Training  of the model. We use 20 epochs**
# 

# In[ ]:


# Training of the model. We use 20 epochs.
history = model.fit(X_TRAIN,Y_TRAIN,epochs=20,validation_split=0.2)


# In[ ]:


Y_PROB = model.predict(X_TEST)


# In[ ]:


Y_PRED = Y_PROB.argmax(axis=1)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_TEST,Y_PRED)


# In[ ]:


# Plot and label the training and validation loss values

plt.plot(history.history['loss'],label='Training')
plt.plot(history.history['val_loss'],label='Validation')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:



# Plot and label the training and validation accuracy values

plt.plot(history.history['accuracy'],label='Training')
plt.plot(history.history['val_accuracy'],label='Validation' )
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


plt.imshow(X_TEST[98])


# In[ ]:


model.predict(X_TEST[98].reshape(1,28,28)).argmax(axis=1)

