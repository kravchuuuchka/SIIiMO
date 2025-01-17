import streamlit as st
import json
import pickle
import sklearn
import pandas as pd
import tensorflow as tf

st.sidebar.header("Введите значения:")
temp = st.sidebar.number_input('Температура (в Цельсиях):', min_value=-40.0, max_value=40.0, value=0.0, step=0.01)
humidity = st.sidebar.number_input('Влажность:', min_value=0.0, max_value=100.0, value=20.0, step=0.01)
hour = st.sidebar.number_input('Время:', min_value=0.0, max_value=23.0, value=12.0, step=1.0)
year = st.sidebar.selectbox('Год:', [0, 1])
df = pd.DataFrame([[temp, humidity, hour, year]])
df.columns = (['temp','humidity','hour','year'])
st.write("Введите не нормализованные значения")
st.write(df)
cl1, cl2 = st.columns(2)
m1_file = open(r"C:\labAI\m1.json", "r")
m1_dist = json.load(m1_file)
m1_file.close()
m2_file = open(r"C:\labAI\m2.json", "r")
m2_dist = json.load(m2_file)
m2_file.close()
def Predict(dictt, values):
    mN = dictt["modelName"]
    r2 = dictt["R2"]
    rmse = dictt["RMSE"]
    
    if mN == "m2.dump":
        m2 = tf.keras.models.load_model(r"C:\labAI\m2.keras")
        snx_file = open(r"C:\labAI\scalerNormForX.dump", "rb")
        snx = pickle.load(snx_file)
        snx_file.close()
        
        sny_file = open(r"C:\labAI\scalerNormForY.dump", "rb")
        sny = pickle.load(sny_file)
        sny_file.close()
        
        cl2.subheader("Нейронная сеть")
        cl2.write("R2 = " + str(r2))
        cl2.write("RMSE = " + str(rmse))
        cl2.write("Данные в нормализованном виде:")
        
        predm2 = pd.DataFrame(m2.predict(snx.transform([values])))
        predm2.columns = ['count']
        cl2.write(predm2)
        
        cl2.write("Полученные данные в не нормализованном виде:")
        predm2init = pd.DataFrame(sny.inverse_transform([[predm2.iloc[0, 0]]]))
        predm2init.columns = ['count']
        cl2.write(predm2init)
    else:
        m1_out = open(r"C:\labAI\m1.dump", "rb")
        m1 = pickle.load(m1_out)
        m1_out.close()
        
        cl1.subheader("Линейная регрессия")
        cl1.write("R2 = " + str(r2))
        cl1.write("RMSE = " + str(rmse))
        

        predm1 = pd.DataFrame(m1.predict(pd.DataFrame([values], columns=['temp', 'humidity', 'hour'])))
        cl1.write("Полученные данные в не нормализованном виде:")
        cl1.write(predm1)

Predict(m1_dist, [temp, humidity, hour])
Predict(m2_dist, [temp, humidity, hour, year])
