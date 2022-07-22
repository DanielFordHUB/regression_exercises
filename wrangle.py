import pandas as pd
import os
from env import get_db_url
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer



def get_zillow_data():
    """Seeks to read the cached zillow.csv first """
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_new_zillow_data()



def get_new_zillow_data():
    '''this function gathers selected data from the ZILLOW SQL DF
    and uses the get_db_url function to connect to said dataframe'''
    sql = '''
    SELECT 
        bedroomcnt AS bedrooms, 
        bathroomcnt AS bathrooms,
        calculatedfinishedsquarefeet AS sq_ft,
        taxvaluedollarcnt AS tax_value,
        yearbuilt AS year_built,
        taxamount AS tax_amnt,
        fips
    FROM
        properties_2017
    JOIN propertylandusetype using (propertylandusetypeid)
    WHERE propertylandusedesc = "Single Family Residential"
    '''
    return pd.read_sql(sql, get_db_url('zillow'))


def handle_nulls(df):    
    # We keep 99.41% of the data after dropping nulls
    # round(df.dropna().shape[0] / df.shape[0], 4) returned .9941
    df = df.dropna()
    return df


def optimize_types(df):
    # Convert some columns to integers
    # fips, yearbuilt, and bedrooms can be integers
    df["fips"] = df["fips"].astype(int)
    df["year_built"] = df["year_built"].astype(int)
    df["bedrooms"] = df["bedrooms"].astype(int)    
    df["tax_value"] = df["tax_value"].astype(int)
    df["sq_ft"] = df["sq_ft"].astype(int)
    return df


def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[df.bathrooms <= 6]
    
    df = df[df.bedrooms <= 6]

    df = df[df.tax_value < 2_000_000]

    return df


def wrangle_zillow():
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    returns a clean dataframe
    """
    df = get_zillow_data()

    df = handle_nulls(df)

    df = optimize_types(df)

    df = handle_outliers(df)

    df.to_csv("zillow.csv", index=False)

    return df




def train_test_validate_split(df, test_size=.2, validate_size=.3, random_state=99):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.
    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=99)
    train, validate = train_test_split(train, test_size=.3, random_state=99)
    
    print(f'train\t n = {train.shape[0]}')
    print(f'test\t n = {test.shape[0]}')
    print(f'validate n = {validate.shape[0]}')
    
    return train, test, validate


## TODO Encode categorical variables (and FIPS is a category so Fips to string to one-hot-encoding
## TODO Scale numeric columns
## TODO Add train/validate/test split in here
## TODO How to handle 0 bedroom, 0 bathroom homes? Drop them? How many? They're probably clerical nulls
