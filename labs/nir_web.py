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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

mns = MinMaxScaler()
sns.set(style="ticks")


@st.cache
def load_data():
    dataset = pd.read_csv('titanik.csv', sep=",")
    dataset = dataset[['Pclass', 'SibSp', 'Parch', 'Survived', 'Age', 'Name', 'Fare']]
    return dataset


@st.cache
def clean_data(dataset):
    data_out = dataset.copy()
    mean_age_miss = data_out[data_out["Name"].str.contains('Miss.', na=False)]['Age'].mean().round()
    mean_age_mrs = data_out[data_out["Name"].str.contains('Mrs.', na=False)]['Age'].mean().round()
    mean_age_mr = data_out[data_out["Name"].str.contains('Mr.', na=False)]['Age'].mean().round()
    mean_age_master = data_out[data_out["Name"].str.contains('Master.', na=False)]['Age'].mean().round()

    print('Mean age of Miss. {}'.format(mean_age_miss))
    print('Mean age of Mrs. {}'.format(mean_age_mrs))
    print('Mean age of Mr. {}'.format(mean_age_mr))
    print('Mean age of Master. {}'.format(mean_age_master))

    def fill_age(name_age):
        
        name = name_age[0]
        age = name_age[1]
        
        if pd.isnull(age):
            if 'Mr.' in name:
                return mean_age_mr
            if 'Mrs.' in name:
                return mean_age_mrs
            if 'Miss.' in name:
                return mean_age_miss
            if 'Master.' in name:
                return mean_age_master
            if 'Dr.' in name:
                return mean_age_master
            if 'Ms.' in name:
                return mean_age_miss
        else:
            return age
    data_out['Age'] = data_out[['Name', 'Age']].apply(fill_age ,axis=1)
    fare_mean_c3 = data_out.Fare[data_out.Pclass == 3].mean()
    data_out['Fare'].fillna(value=fare_mean_c3, inplace=True)
    data_out.drop(['Name'], axis=1, inplace=True)
    return data_out

st.sidebar.header('Метод k-ближайших соседей')
data = load_data()
data = clean_data(data)

cv_slider = st.sidebar.slider('Значение k для модели:', min_value=3, max_value=20, value=3, step=1)

st.subheader('Первые 5 значений')
st.write(data.head(5))
st.write(data.shape)

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)


X = data.drop(['Survived'], axis=1)
y = data['Survived']
X = pd.DataFrame(mns.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)

knn = KNeighborsClassifier(n_neighbors=cv_slider)
knn.fit(X_train, y_train)
predict_values = knn.predict(X_test)
predict_dataset = X_test
predict_dataset['Survived'] = predict_values

st.subheader('Выполнение алгоритма с k = ' + str(cv_slider))
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(ax=ax, x="Age", y="Survived", data=predict_dataset)
st.pyplot(fig)

st.subheader('Оценка качества модели')
fig, ax = plt.subplots(figsize=(10,5))
cm = confusion_matrix(y_test, predict_values)
sns.heatmap(cm, annot=True)
plt.xlabel('Predict values')
plt.ylabel('True values')
st.pyplot(fig)