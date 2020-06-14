#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk 
from tkinter import messagebox,simpledialog,filedialog
from tkinter import *
import tkinter
from imutils import paths
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


# In[2]:


root= tk.Tk() 
root.title("Indian Air Quality Analysis")
root.geometry("1300x1200")


# In[3]:


def upload_data():
    global aqi
    aqi= askopenfilename(initialdir = "Dataset")
    #pathlabel.config(text=train_data)
    text.insert(END,"Dataset loaded\n\n")
    


# In[4]:


def data():
    global aqi
    text.delete('1.0',END)
    aqi= pd.read_csv('data.csv', encoding = 'unicode_escape',)
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,aqi.head())
    text.insert(END,"column names\n\n")
    text.insert(END,aqi.columns)
    text.insert(END,"Total no. of rows and coulmns\n\n")
    text.insert(END,aqi.shape)
    
    


# In[5]:


def statistics():
    text.delete('1.0',END)
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,aqi.head())
    stats=aqi.describe()
    text.insert(END,"\n\nStatistical Measurements for Data\n\n")
    text.insert(END,stats)
    null=aqi.isnull().sum()
    text.insert(END,null)


# In[9]:


def preprocess():
    from sklearn.preprocessing import Imputer
    global aqi,df
    text.delete('1.0',END)
    aqi.drop(['stn_code', 'agency', 'sampling_date', 'location_monitoring_station'], axis=1, inplace=True)
    aqi = aqi.dropna(subset=['date'])
    aqi.state = aqi.state.replace({'Uttaranchal':'Uttarakhand'})
    aqi.state[aqi.location == "Jamshedpur"] = aqi.state[aqi.location == 'Jamshedpur'].replace({"Bihar":"Jharkhand"})
    types = {
    "Residential": "R",
    "Residential and others": "RO",
    "Residential, Rural and other Areas": "RRO",
    "Industrial Area": "I",
    "Industrial Areas": "I",
    "Industrial": "I",
    "Sensitive Area": "S",
    "Sensitive Areas": "S",
    "Sensitive": "S",
    np.nan: "RRO"
    }
    aqi.type = aqi.type.replace(types)
    VALUE_COLS = ['so2', 'no2', 'rspm', 'spm', 'pm2_5']
    imputer = Imputer(missing_values=np.nan, strategy='mean')
    aqi[VALUE_COLS] = imputer.fit_transform(aqi[VALUE_COLS])
    null=aqi.isnull().sum()
    def calculate_si(so2):
        si=0
        if (so2<=40):
            si= so2*(50/40)
        if (so2>40 and so2<=80):
            si= 50+(so2-40)*(50/40)
        if (so2>80 and so2<=380):
            si= 100+(so2-80)*(100/300)
        if (so2>380 and so2<=800):
            si= 200+(so2-380)*(100/800)
        if (so2>800 and so2<=1600):
            si= 300+(so2-800)*(100/800)
        if (so2>1600):
            si= 400+(so2-1600)*(100/800)
        return si
    aqi['si']=aqi['so2'].apply(calculate_si)
    df= aqi[['so2','si']]
    def calculate_ni(no2):
        ni=0
        if(no2<=40):
            ni= no2*50/40
        elif(no2>40 and no2<=80):
            ni= 50+(no2-14)*(50/40)
        elif(no2>80 and no2<=180):
            ni= 100+(no2-80)*(100/100)
        elif(no2>180 and no2<=280):
            ni= 200+(no2-180)*(100/100)
        elif(no2>280 and no2<=400):
            ni= 300+(no2-280)*(100/120)
        else:
            ni= 400+(no2-400)*(100/120)
        return ni
    aqi['ni']=aqi['no2'].apply(calculate_ni)
    def calculate_rpi(rspm):
        rpi=0
        if(rpi<=30):
            rpi=rpi*50/30
        elif(rpi>30 and rpi<=60):
            rpi=50+(rpi-30)*50/30
        elif(rpi>60 and rpi<=90):
            rpi=100+(rpi-60)*100/30
        elif(rpi>90 and rpi<=120):
            rpi=200+(rpi-90)*100/30
        elif(rpi>120 and rpi<=250):
            rpi=300+(rpi-120)*(100/130)
        else:
            rpi=400+(rpi-250)*(100/130)
        return rpi
    aqi['rpi']=aqi['rspm'].apply(calculate_rpi)
    def calculate_spi(spm):
        spi=0
        if(spm<=50):
            spi=spm
        if(spm<50 and spm<=100):
            spi=spm
        elif(spm>100 and spm<=250):
            spi= 100+(spm-100)*(100/150)
        elif(spm>250 and spm<=350):
            spi=200+(spm-250)
        elif(spm>350 and spm<=450):
            spi=300+(spm-350)*(100/80)
        else:
            spi=400+(spm-430)*(100/80)
        return spi
    aqi['spi']=aqi['spm'].apply(calculate_spi)
    df= aqi[['so2','si','no2','ni','rspm','rpi','spm','spi']]
    def calculate_aqi(si,ni,spi,rpi):
        aqi=0
        if(si>ni and si>spi and si>rpi):
            aqi=si
        if(spi>si and spi>ni and spi>rpi):
            aqi=spi
        if(ni>si and ni>spi and ni>rpi):
            aqi=ni
        if(rpi>si and rpi>ni and rpi>spi):
            aqi=rpi
        return aqi
    aqi['AQI']=aqi.apply(lambda aqi:calculate_aqi(aqi['si'],aqi['ni'],aqi['spi'],aqi['rpi']),axis=1)
    df= aqi[['date','state','si','ni','rpi','spi','AQI']]
    df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
    df['year'] = df['date'].dt.year # year
    df['year'] = df['year'].fillna(0.0).astype(int)
    df.drop(['date','state'], axis=1, inplace=True)
    text.insert(END,df.head())
    return df
    
    


# In[10]:


def train_test():
    text.delete('1.0',END)
    global X,y
    global x_train,x_test,y_train,y_test,X_train,X_test
    text.delete('1.0',END)
    X=df.drop(['AQI'],axis=1)
    y=df['AQI']
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)
    from sklearn.preprocessing import MinMaxScaler
    sc_X = MinMaxScaler()
    x_train = sc_X.fit_transform(X_train)
    x_test = sc_X.transform(X_test)
    text.insert(END,"Train and Test model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(df))+"\n")
    text.insert(END,"Training Size : "+str(len(x_train))+"\n")
    text.insert(END,"Test Size : "+str(len(x_test))+"\n")
    return x_train,x_test,y_train,y_test,X_train,X_test
    


# In[12]:


def RF():
    global New_data,data_test
    global x_train,x_test,y_train,y_test
    text.delete('1.0',END)
    clf = RandomForestRegressor(n_estimators=50, max_features='sqrt')
    clf = clf.fit(x_train, y_train)
    
    predictions = clf.predict(x_test)
    df_output = pd.DataFrame()
    df_output['AQI'] = df['AQI']
    df_output['AQI Predicted'] = pd.DataFrame(predictions)
    df_output[['AQI','AQI Predicted']].to_csv('AQI@RF.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")


# In[13]:


def LR():
    text.delete('1.0',END)
    lr = LinearRegression()
    lr = lr.fit(x_train, y_train)
    
    predictions = lr.predict(x_test)
    df_output = pd.DataFrame()
    df_output['AQI'] = df['AQI']
    df_output['AQI Predicted'] = pd.DataFrame(predictions)
    df_output[['AQI','AQI Predicted']].to_csv('AQI@LR.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")


# In[14]:


def KNN():
    text.delete('1.0',END)
    knn = KNeighborsRegressor()
    knn = knn.fit(x_train, y_train)
    
    predictions = knn.predict(x_test)
    df_output = pd.DataFrame()
    df_output['AQI'] = df['AQI']
    df_output['AQI Predicted'] = pd.DataFrame(predictions)
    df_output[['AQI','AQI Predicted']].to_csv('AQI@KNN.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")

    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")


# In[15]:


def SVR():
    text.delete('1.0',END)
    svr = SVR()
    svr = svr.fit(x_train, y_train)
    
    predictions = svr.predict(x_test)
    df_output = pd.DataFrame()
    df_output['AQI'] = df['AQI']
    df_output['AQI Predicted'] = pd.DataFrame(predictions)
    df_output[['AQI','AQI Predicted']].to_csv('AQI@SVR.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")


# In[16]:


def input_values():
    text.delete('1.0',END)
    global new_x_train,new_x_test
    global RFT
     
    
    global si #our 2nd input variable
    si = float(entry1.get())

    global ni 
    ni = float(entry2.get())

    global rpi
    rpi = float(entry3.get())
    
    global spi
    spi = float(entry4.get())
    
    global year
    year = float(entry5.get())
    
    list=[[si,ni,rpi,spi,year]]
    parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

    rf = RandomForestRegressor(**parameters)
    rf.fit(x_train, y_train)
    Prediction_result  = rf.predict(list)
    text.insert(END,"New values are predicted from Random Forest Regressor\n\n")
    text.insert(END,"Predicted AQI for the New inputs\n\n")
    text.insert(END,Prediction_result)


# In[ ]:
font = ('times', 14, 'bold')
title = Label(root, text='Indian Air Quality Analysis')  
title.config(font=font)           
title.config(height=2, width=120)       
title.place(x=0,y=5)

font1 = ('times',13 ,'bold')
button1 = tk.Button (root, text='Upload Data1',width=13,command=upload_data) 
button1.config(font=font1)
button1.place(x=60,y=100)

button2 = tk.Button (root, text='Data',width=13,command=data)
button2.config(font=font1)
button2.place(x=60,y=150)

button3 = tk.Button (root, text='statistics',width=13,command=statistics)  
button3.config(font=font1)
button3.place(x=60,y=200)


button4 = tk.Button (root, text='preprocess',width=13,command=preprocess)
button4.config(font=font1) 
button4.place(x=60,y=250)

button5 = tk.Button (root, text='Train & Test',width=13,command=train_test)
button5.config(font=font1) 
button5.place(x=60,y=300)

title = Label(root, text='Application of ML models')
#title.config(bg='RoyalBlue2', fg='white')  
title.config(font=font1)           
title.config(width=25)       
title.place(x=250,y=70)

button6 = tk.Button (root, text='RFT',width=15,bg='pale green',command=RF)
button6.config(font=font1) 
button6.place(x=300,y=100)

button7 = tk.Button (root, text='LR',width=15,bg='sky blue',command=LR)
button7.config(font=font1) 
button7.place(x=300,y=150)

button8 = tk.Button (root, text='KNN',width=15,bg='orange',command=KNN)
button8.config(font=font1) 
button8.place(x=300,y=200)

button9 = tk.Button (root, text='SVR',width=15,bg='violet',command=SVR)
button9.config(font=font1) 
button9.place(x=300,y=250)



title = Label(root, text='Enter Input values for the New Prediction')
title.config(bg='black', fg='white')  
title.config(font=font1)           
title.config(width=40)       
title.place(x=60,y=380)

font3=('times',9,'bold')
title1 = Label(root, text='*You Should enter scaled values between 0 and 1')
 
title1.config(font=font3)           
title1.config(width=40)       
title1.place(x=50,y=415)

def clear1(event):
    entry1.delete(0, tk.END)

font2=('times',10)
entry1 = tk.Entry (root) # create 1st entry box
entry1.config(font=font2)
entry1.place(x=60, y=450,height=30,width=150)
entry1.insert(0,'si')
entry1.bind("<FocusIn>",clear1)

def clear2(event):
    entry2.delete(0, tk.END)

font2=('times',10)
entry2 = tk.Entry (root) # create 1st entry box
entry2.config(font=font2)
entry2.place(x=315, y=450,height=30,width=150)
entry2.insert(0,'ni')
entry2.bind("<FocusIn>",clear2)


def clear3(event):
    entry3.delete(0, tk.END)

font2=('times',10)
entry3 = tk.Entry (root) # create 1st entry box
entry3.config(font=font2)
entry3.place(x=60, y=500,height=30,width=150)
entry3.insert(0,'rpi')
entry3.bind("<FocusIn>",clear3)

def clear4(event):
    entry4.delete(0, tk.END)

font2=('times',10)
entry4 = tk.Entry (root) # create 1st entry box
entry4.config(font=font2)
entry4.place(x=315, y=500,height=30,width=150)
entry4.insert(0,'spi')
entry4.bind("<FocusIn>",clear4)

def clear5(event):
    entry5.delete(0, tk.END)

font2=('times',10)
entry5 = tk.Entry (root) # create 1st entry box
entry5.config(font=font2)
entry5.place(x=60, y=550,height=30,width=150)
entry5.insert(0,'year')
entry5.bind("<FocusIn>",clear5)



Prediction = tk.Button (root, text='Prediction',width=15,fg='white',bg='green',command=input_values)
Prediction.config(font=font1) 
Prediction.place(x=180,y=600)



font1 = ('times', 11, 'bold')
text=Text(root,height=32,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set,xscrollcommand=scroll.set)
text.place(x=550,y=70)
text.config(font=font1)

root.mainloop()






