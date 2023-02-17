#!/usr/bin/env python
# coding: utf-8

# # SOUND SIGNAL CLASSIFICATION

# In[89]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 


# In[90]:


our_source_file_destination='UrbanSound8K/17973-2-0-32.wav'
librosa_audio_data, librosa_sample_rate = librosa.load(our_source_file_destination)


# In[91]:


print(librosa_audio_data)


# In[92]:


plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)
plt.show()


# In[93]:


mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40) 
print(mfccs.shape)


# In[94]:


mfccs


# In[95]:


audio_dataset_path='UrbanSound8K/audio/'
metadata=pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
metadata.head()


# In[96]:


def features_extractor(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features


# In[97]:


extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])


# In[98]:


extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()


# In[99]:


X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[100]:


labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


# In[101]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[102]:


num_labels = 10


# In[103]:


model=Sequential()

model.add(Dense(125,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(125))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[104]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[105]:


epochscount = 350
num_batch_size = 32

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=epochscount, validation_data=(X_test, y_test), verbose=1)


# In[106]:


validation_test_set_accuracy = model.evaluate(X_test,y_test,verbose=0)
print("%",validation_test_set_accuracy[1]*100)


# In[107]:


model.predict_classes(X_test)


# In[108]:


filename="UrbanSound8K/17973-2-0-32.wav"
sound_signal, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=sound_signal, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)


# In[109]:


print(mfccs_scaled_features)


# In[110]:


mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)


# In[111]:


print(mfccs_scaled_features.shape)


# In[112]:


our_classes = model.predict(mfccs_scaled_features)


# In[113]:


our_classes


# In[114]:


dataset_classes = ["Air Conditioner","Car Horn","Children Playing Sound","Dog Bark","Drilling", "Engine Idling", "Gun Shot", "Jackhammer", "Siren Sound", "Street Music Sound"]

result = np.argmax(our_classes[0])
print(dataset_classes[result]) 


our_source_file_destination='UrbanSound8K/17973-2-0-32.wav'
librosa_audio_data, librosa_sample_rate = librosa.load(our_source_file_destination)
plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)
plt.show()


# In[ ]:





# In[ ]:




