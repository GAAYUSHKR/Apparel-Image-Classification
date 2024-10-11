
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm


# In[2]:


DATADIR='C:/Users/Avinash/Downloads/train_LbELtWX/train'


# In[3]:


category=pd.read_csv("file:///C:/Users/Avinash/Downloads/train_LbELtWX/train.csv")


# In[4]:


labels=np.array(category['label'])
print(labels)


# In[ ]:


path = os.path.join(DATADIR)
train=[]
i=0

for img in tqdm(os.listdir(path)):
    try:
        img_array = cv2.imread(os.path.join(path,img))
        train.append([img_array])
        i=i+1
    except Exception as e:
            pass 


# In[6]:


len(train)


# In[12]:


for i in range(5):
    plt.imshow(train[i])
    plt.show()


# In[13]:


y=labels
print(y)


# In[14]:


y=np.array(y)


# In[18]:


import pickle

pickle_out=open("x.pickle","wb")
pickle.dump(train,pickle_out)
pickle_out.close()


pickle_out=open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

