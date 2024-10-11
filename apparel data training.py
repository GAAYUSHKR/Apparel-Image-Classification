
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


import pickle

pickle_in=open("x.pickle","rb")
x=pickle.load(pickle_in)


pickle_in=open("y.pickle","rb")
y=pickle.load(pickle_in)


# In[18]:


y


# In[70]:


plt.imshow(x[235])
plt.show()
x.shape
y[100]


# In[16]:


w_grid=15
l_grid=15

fig,axes=plt.subplots(l_grid,w_grid,figsize=(17,17))

axes=axes.ravel()
n=len(y)

for i in np.arange(0,225):
    index=np.random.randint(0,n)
    
    axes[i].imshow( x[index] )
    axes[i].set_title(y[index], fontsize = 8)
    axes[i].axis('off')
    


# In[37]:


x[1]


# In[25]:


x=x/255


# In[40]:


from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.3)


# In[41]:


x_train.shape


# In[43]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam


# In[68]:


model = Sequential()

model.add(Conv2D(32,3, 3, input_shape = (28,28,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))

model.add(Dropout(0.25))

model.add(Conv2D(64,3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))

model.add(Flatten()) 

model.add(Dense(output_dim = 32, activation = 'relu'))
model.add(Dense(output_dim = 10, activation = 'sigmoid'))

model.compile(loss ='sparse_categorical_crossentropy', optimizer='sgd',metrics =['accuracy'])


# In[69]:


epochs = 5

cnn = model.fit(x_train,
                        y_train,
                        batch_size = 512,
                        nb_epoch = epochs,
                        verbose = 1,
                        validation_data = (x_valid, y_valid))

