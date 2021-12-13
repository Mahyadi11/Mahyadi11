#!/usr/bin/env python
# coding: utf-8

# In[1]:


#library
import pandas as pd
import numpy as np
import joblib 
from sklearn.linear_model import LinearRegression


# In[2]:


#dataset 
df = pd.read_csv("D:/coobaa.csv")


# In[3]:


#Library membuat model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,r2_score
from sklearn.model_selection import train_test_split


# In[5]:


X=df.iloc[:,2:8].values
y=df.iloc[:,1].values


# In[6]:


#Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[7]:


#call model regression
X = np.asanyarray(df[['Luas_hutan']])
Y = np.asanyarray(df[['Intensitas_em']])
model = LinearRegression().fit(X,Y)
model


# In[8]:


#save model
filename = 'model.sav'
joblib.dump(model, filename)


# In[9]:


#load model
loaded_model = joblib.load(filename)


# In[10]:


#prediction model
loaded_model.predict(np.array([10]).reshape(1, 1))


# In[12]:


from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__, template_folder='templates')
@app.route('/')
def student():
   return render_template("home.html")
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1,1)
    loaded_model = joblib.load('model.sav')
    result = loaded_model.predict(to_predict)
    return result[0]
@app.route('/',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
    to_predict_list = request.form.to_dict()
    to_predict_list=list(to_predict_list.values())
    to_predict_list = list(map(float, to_predict_list))
    result = float(ValuePredictor(to_predict_list))
    return render_template("home.html",result = result)
if __name__ == '__main__':
   app.run(debug = True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




