# base
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import streamlit as st

# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import GridSearchCV, KFold

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

#models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import root_mean_squared_log_error

import category_encoders as ce 
# tunning hyperparamters model
import optuna

#Data loading
default_dataset = 'Data/test.csv'
ml_pipeline_LG = pickle.load(open('pipl.pkl', 'rb'))
ml_pipeline_KNN = pickle.load(open('pipl_KNN.pkl', 'rb'))


st.header('Предсказание цены на недвижимость')
st.sidebar.header('Загрузите пользовательский файл с параметрами')
uploaded_file = st.sidebar.file_uploader("Upload your dataset",['cvs'])

if uploaded_file is not None:
    file = uploaded_file
else:
    file = default_dataset

try:
    df = pd.read_csv(file)
except Exception as err:
    st.error(f'An error occurred while cvs reading: {err}', icon="🚨")

st.subheader('5 строк датасета')
st.dataframe(df.head(5))

st.subheader('На основании train датасета было обучено 3 модели: линейной регрессии, KNN регрессии и CatBoost регресии')

if st.button('Получить предсказания по тестовым данным'):
    y_pred = pd.Series(ml_pipeline_LG.predict(df))
    st.subheader('Метод линейной регрессии:')
    st.dataframe(y_pred.to_frame().rename(columns={0:'Sales_pred'}).T)

    y_pred = pd.Series(ml_pipeline_KNN.predict(df))
    st.subheader('Метод ближайших соседей:')
    st.dataframe(y_pred.to_frame().rename(columns={0:'Sales_pred'}).T)
    # st.subheader('Полученная метрика:')
    # st.write(f'{y_pred}')

# imput = pd.DataFrame()
# st.dateframe(imput)

# NaNs_only = st.checkbox('Show only columns with NaNs',value=False)

# def summary(NaNs_only):
#     #Summary info block
#     st.subheader('Summary information about Dataset')
#     rows, cols = df.shape
#     st.write(f'Dataset size: {rows} rows, {cols} columns')
#     summary = pd.DataFrame(data={'NaN_count': df.isna().sum(), 'data_type':df.dtypes})
#     if NaNs_only:
#         st.dataframe(summary[summary['NaN_count'] != 0].T)
#     else:
#         st.dataframe(summary.T)

# summary(NaNs_only)

# #First analysis block for distribution of sample

# #Idea: filtering by dataset columns to show describe info
# option = st.selectbox(
#    "Choose column to see summary information",
#    df.columns,
#    index=None,
#    placeholder="Select column...",
# )

# st.write(f"You selected: {option} with type {type(option)}")

# if st.button('Start preprocessing column'):
#     if option is None:
#         st.error('WTF')
#     else:
#         st.dataframe(df[option].describe())