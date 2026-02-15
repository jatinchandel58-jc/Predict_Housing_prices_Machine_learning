import os
import joblib

import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder ,StandardScaler 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model_pkl"
PIPELINE_FILE = "pipeline_pkl"

def build_pipeline(num_attribut,cat_attribut):
    num_pipeling = Pipeline([
    ("impute",SimpleImputer(strategy='median')),
    ("scaler",StandardScaler())
    ])

    cat_pipeling = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeling = ColumnTransformer([
        ("num",num_pipeling,num_attribut),
        ("cat",cat_pipeling,cat_attribut),
    ])

    return full_pipeling

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("housing.csv")

    #
    df["income_cat"] = pd.cut(df["median_income"],
                            bins=[0,1.5,3.0,4.5,6.0,np.inf],
                            labels= [1,2,3,4,5])

    #
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(df,df["income_cat"]):
        df.loc[test_index].drop(["income_cat"],axis=1).to_csv("input.csv",index=False)
        housing = df.loc[train_index].drop(["income_cat"],axis=1)
    
    # feature and labels 
    housing_labels = housing['median_house_value'].copy()
    housing_feature = housing.drop(['median_house_value'],axis=1)

    # num_values and cat_values
    num_attribut = housing_feature.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribut = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribut,cat_attribut)
    housing_preperd = pipeline.fit_transform(housing_feature)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_preperd,housing_labels)

    #
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print("Model trainning saves")

else:

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transform_input = pipeline.transform(input_data)
    prediction = model.predict(transform_input)
    input_data["median_house_value"]= prediction

    #
    input_data.to_csv("output.csv",index=False)
    print("inferance is complete ! Thankyou ")



    

















