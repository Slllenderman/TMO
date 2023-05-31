import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix
sns.set(style="ticks")


@st.cache
def load_data():
    dataset = pd.read_csv('iris.csv', sep=",", nrows=500)
    return dataset


@st.cache
def ordinal_encoder(dataset, column):
    data_out = dataset.copy()
    ord_enc = OrdinalEncoder()
    new_column = ord_enc.fit_transform(data_out[[column]])
    data_out = data_out.drop(columns=[column], axis=1)
    data_out[column] = new_column
    return data_out

st.sidebar.header('Метод k-ближайших соседей')
data = load_data()
cv_slider = st.sidebar.slider('Значение k для модели:', min_value=3, max_value=20, value=3, step=1)

st.subheader('Первые 5 значений')
st.write(data.head(5))
st.write(data.shape)

st.subheader('Уникальные значения столбца Species')
st.write(data["Species"].unique())

data = ordinal_encoder(data, "Species")
st.subheader('Кодирование категориальных признаков')
st.write(data.head(5))

st.subheader('Уникальные значения столбца Species')
st.write(data["Species"].unique())

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)

data_train, data_test = train_test_split(data, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=cv_slider)
knn.fit(data_train[['SepalLengthCm', 'PetalLengthCm']], data_train[['Species']])
predict_values = knn.predict(data_test[['SepalLengthCm', 'PetalLengthCm']])
predict_dataset = data_test[['SepalLengthCm', 'PetalLengthCm']]
predict_dataset['Species'] = predict_values

st.subheader('Выполнение алгоритма с k = ' + str(cv_slider))
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(ax=ax, x="SepalLengthCm", y="PetalLengthCm", data=predict_dataset, hue="Species")
st.pyplot(fig)

st.subheader('Оценка качества модели')
fig, ax = plt.subplots(figsize=(10,5))
cm = confusion_matrix(data_test[["Species"]], predict_values)
sns.heatmap(cm, annot=True)
plt.xlabel('Predict values')
plt.ylabel('True values')
st.pyplot(fig)