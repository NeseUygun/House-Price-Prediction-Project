#####################################################
# HOUSE PRICE PREDICTION
#####################################################


# Required Library and Functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Veri setinin okutulması
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# train and test setlerini birleştiriyoruz
df = train.append(test).reset_index(drop=True)

######################################
# EDA
######################################


######################################
# Overall Picture
######################################

check_df(df)

# Removing outliers from data
df = df.loc[df["SalePrice"]<=400000,]

check_df(df)

##################################
# Capture numeric and categorical variables.
##################################

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

######################################
# Analysis of categorical variables.
######################################

for col in cat_cols:
    cat_summary(df, col)


######################################
# Analysis of Numerical Variables
######################################

for col in num_cols:
    num_summary(df, col, True)

######################################
# Analysis of Target Variable
######################################

for col in cat_cols:
    target_summary_with_cat(df,"SalePrice",col)

######################################
# Analysis of Correlation
######################################

corr_matrix = df.corr()

# Show correlation
threshold = 0.5
filtre = np.abs(corr_matrix["SalePrice"]) > threshold
corr_features=corr_matrix.columns[filtre].tolist()
sns.clustermap (df[corr_features].corr(), annot = True, fmt = ".2f",cmap = "viridis")
plt.title("Correlation Between Features w/ Corr Threshold 0.75")
plt.show()


######################################
# Data Preprocessing
######################################

######################################
# Analysis outlier values
######################################

# Outlier check
for col in num_cols:
    if col != "SalePrice":
      print(col, check_outlier(df, col))

# Outlier suppression process
for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)


######################################
# Analysis of missing values
######################################

missing_values_table(df)
# ratio: Returns the percentage of missing observations in the column
# n_miss: Returns the number of missing observations in each column

### PoolQC, MiscFeature, Alley -> useless kolona eklenecek


# Null values in some variables indicate that the house does not have that feature.
no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
for col in no_cols:
    df[col].fillna("No",inplace=True)

missing_values_table(df)

# Allows filling missing values with median or mean

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Variables with missing values are listed

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Number of missing values of variables before implementation

    # Fill nulls with mode if variable is object and number of classes is less than or equal to cat_length
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # If num_method is mean, the null values of non-object type variables are filled with the mean
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # If num_method is median, the null values of non-object type variables are filled with the mean
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)
df.columns

df["SalePrice"].mean() # 174873.43854748603
df["SalePrice"].std() # 65922.70393689284

######################################
# RARE
######################################

# Examining the distribution of categorical columns

rare_analyser(df, "SalePrice", cat_cols)

# Detection of rare classes
df = rare_encoder(df, 0.01, cat_cols)

rare_analyser(df, "SalePrice", cat_cols)


useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]
# ['Street', 'Utilities', 'PoolQC', 'PoolArea'] -> these will be added useless_col


useless_cols = ['Street', 'Utilities', 'PoolQC', 'PoolArea',"MiscFeature","Alley","LandContour","LandSlope", 'Neighborhood']


df.drop(useless_cols, axis=1, inplace=True)



######################################
# Feature Engineering
######################################

# 1stFlrSF: Birinci Kat metre kare alanı
# GrLivArea: Üstü (zemin) oturma alanı metre karesi
df["new_1st_GrLiv"] = df["1stFlrSF"]/df["GrLivArea"]
df["new_Garage_GrLiv"] = df["GarageArea"]/df["GrLivArea"]

# GarageQual
# GarageCond
df["GarageQual"].value_counts()
df["GarageCond"].value_counts()
df["TotalGarageQual"] = df[["GarageQual", "GarageCond"]].sum(axis = 1)

# LotShape
df["LotShape"].value_counts()
df.loc[(df["LotShape"] == "IR2"), "LotShape"] = "IR1"
df.loc[(df["LotShape"] == "IR3"), "LotShape"] = "IR1"

# Total bath number
# BsmtFullBath
# BsmtHalfBath
# HalfBath
df["new_total_bath"] = df["BsmtFullBath"] + df["BsmtHalfBath"] + df["HalfBath"]

# Building age
# YearBuilt
# YearRemodAdd
df["new_built_remodadd"] =  df["YearRemodAdd"] - df["YearBuilt"]

# YrSold
# YearBuilt
# YearRemodAdd
df["new_HouseAge"] = df.YrSold - df.YearBuilt
df["new_RestorationAge"] = df.YrSold - df.YearRemodAdd
df["new_GarageAge"] = df.GarageYrBlt - df.YearBuilt
df["new_GarageSold"] = df.YrSold - df.GarageYrBlt


# GrLivArea
# LotArea
# TotalBsmtSF
# GrLivArea
df["new_GrLivArea_LotArea"] = df["GrLivArea"] / df["LotArea"]
df["total_living_area"] = df["TotalBsmtSF"] + df["GrLivArea"]


##Total Floor
# 1stFlrSF
# 2ndFlrSF
# TotalBsmtSF
# GrLivArea
df["new_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
df["new_TotalHouseArea"] = df.new_TotalFlrSF + df.TotalBsmtSF
df["new_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF

# Lot Ratio
# GrLivArea
# LotArea
df["new_LotRatio"] = df.GrLivArea / df.LotArea
df["new_RatioArea"] = df.new_TotalHouseArea / df.LotArea
df["new_GarageLotRatio"] = df.GarageArea / df.LotArea


##################
#  One-Hot Encoding
##################

cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


df = one_hot_encoder(df, cat_cols, drop_first=True)


##################################
# Modeling
##################################

y = df['SalePrice']
X = df.drop(["Id", "SalePrice"], axis=1)

# Split data as test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

### BASE MODELS ###

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 135772067.8441 (LR)
# RMSE: 23513.2685 (Ridge)
# RMSE: 24449.7478 (Lasso)
# RMSE: 26869.9511 (ElasticNet)
# RMSE: 36225.8091 (KNN)
# RMSE: 35694.1323 (CART)
# RMSE: 24767.3153 (RF)
# RMSE: 22774.9258 (GBM)
# RMSE: 24659.0742 (XGBoost)
# RMSE: 22581.9479 (LightGBM)

### Hyperparameter Optimization ###
# -> By balancing the model complexity, the balance of overfitting and underfitting can be achieved.


lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1, 0.05],
               "n_estimators": [1500, 3000, 6000], 
               "colsample_bytree": [0.5, 0.7], 
               "num_leaves": [31, 35], 
               "max_depth": [3, 5]} 

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)


rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
# 21842.90302800053


