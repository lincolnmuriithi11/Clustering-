import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire and Clean Data ######################

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
# ----------------------------------------------------------------- #
def get_zillow_sql():#this function acquires zillow data, and converts it to a dataframe

    ''' this function calls a sql file from the codeup database and creates a data frame from the zillow db. use select max transaction
    '''
    query ='''
     SELECT  
    *
    FROM properties_2017 P_2017
    JOIN (
        SELECT parcelid, MAX(transactiondate) AS max_transactiondate
        FROM predictions_2017
        GROUP BY parcelid
    ) pred USING(parcelid)
    JOIN
    predictions_2017 USING (parcelid)
        LEFT JOIN
    propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN
    airconditioningtype USING (airconditioningtypeid)
        LEFT JOIN
    architecturalstyletype USING (architecturalstyletypeid)
        LEFT JOIN
    buildingclasstype USING (buildingclasstypeid)
        LEFT JOIN
    heatingorsystemtype USING (heatingorsystemtypeid)
        LEFT JOIN
    storytype USING (storytypeid)
        LEFT JOIN
    typeconstructiontype USING (typeconstructiontypeid)
        LEFT JOIN
    unique_properties USING (parcelid)
    WHERE
    propertylandusedesc = 'Single Family Residential'
         
        '''
    df = pd.read_sql(query, get_connection('zillow'))
    #creating a df for easy access 
    return df
        
def get_zillow_df():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_cluster.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_cluster.csv')
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = get_zillow_sql()
        
        # Cache data
        df.to_csv('zillow_cluster.csv', index = 0)
        
    return df

def handle_nulls(df, percent_thresh_column, percent_thresh_row):
    n_required_column = round(df.shape[0] * percent_thresh_column)
    n_required_row = round(df.shape[1] * percent_thresh_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1)
    return df




