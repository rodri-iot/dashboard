import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.preprocessing import (MinMaxScaler,
                                   StandardScaler,
                                   LabelEncoder,
                                   OneHotEncoder)
from sklearn.feature_selection import (chi2,
                                       SelectKBest,
                                       f_regression)
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV) # For Optimize
from sklearn.metrics import (accuracy_score,
                             mean_squared_error,
                            confusion_matrix,
                            classification_report)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.naive_bayes import (GaussianNB,
                                 MultinomialNB,
                                 BernoulliNB)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neighbors import KNeighborsRegressor


# Import data

def load_data():
    url = 'https://raw.githubusercontent.com/rfordatascience/' + \
    'tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv'
    df = pd.read_csv(url)
    df_interim = df.copy()
    df_interim = df_interim[['total_cup_points',
                                'species',
                                'country_of_origin',
                                'variety',
                                'aroma',
                                'aftertaste',
                                'acidity',
                                'body',
                                'balance',
                                'sweetness',
                                'altitude_mean_meters',
                                'moisture']]
    df_interim = df_interim.dropna()
    df_interim['species'] = pd.Categorical(df_interim['species'])
    df_interim['country_of_origin'] = pd.Categorical(df_interim['country_of_origin'])
    df_interim['variety'] = pd.Categorical(df_interim['variety'])
    df_interim['specialty'] = df_interim['total_cup_points'].apply(lambda x: 'yes' if x>82.43 else 'no')
    df_interim['specialty'] = pd.Categorical(df_interim['specialty'])
    df_interim['altitude_mean_meters'] = df_interim['altitude_mean_meters'].apply(lambda x:1300 if x>10000 else x)
    df = df_interim.copy()

    return df

df_ch = load_data()
# Shape
st.write(df_ch.shape[0])

# Name dashboard
st.title('Coffee Dashboard')

# Data Frame
st.dataframe(df_ch)

# EDA
st.title('EDA')

# Describe Statistics
st.write('Describe Statisticals - Numbers')
fig0 = df_ch.select_dtypes(include='number').describe().T
st.dataframe(fig0)

st.write('Describe Statistics - Categorical')
fig0_1 = df_ch.select_dtypes(include='category').describe().T
st.dataframe(fig0_1)

# Univariate analysis
st.write('Univariate analysis')
fig, ax = plt.subplots(3,3, figsize=(20,20))

cols = df_ch.select_dtypes(include='number').columns

for i, var in enumerate(cols):
    row = i // 3
    col = i % 3
    if row < 3 and col < 3:  # Asegúrate de no exceder los límites de la cuadrícula
        ax[row, col].hist(df_ch[var].dropna(), bins=20, color='blue', edgecolor='black')
        ax[row, col].set_title(var.capitalize())

st.pyplot(fig)

# Bivariate analysis
st.write('Bivariate analysis')
fig2 = sns.pairplot(df_ch.drop(['species', 'country_of_origin', 'variety'], axis=1), hue='specialty')
st.pyplot(fig2.fig)

# Corr
st.write('Correlation')
fig3 = df_ch.select_dtypes(include='number').corr()
st.dataframe(fig3)

# ML
df_train, df_test = train_test_split(df_ch, test_size=0.2, stratify=df_ch['specialty'])

X_train = df_train.drop([['']])