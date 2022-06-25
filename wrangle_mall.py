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

################################ MALL DATA ######################################







# Acquire data from the customers table in the mall_customers database.

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


#this function acquires mall data, and converts it to a dataframe
def get_mall_sql():

    ''' this function calls a sql file from the codeup database and creates a data frame from the mall db. 
    '''
    query ='''
     SELECT * FROM customers
        '''
    df = pd.read_sql(query, get_connection('mall_customers'))
    #creating a df for easy access 
    return df 

def get_mall_df():
    '''
    This function reads in mall data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('mall.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('mall.csv')
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = get_mall_sql()
        
        # Cache data
        df.to_csv('mall.csv', index = 0)
        
    return df

# Detect outliers using IQR.
def handle_outliers(df, cols, k):
    # Create placeholder dictionary for each columns bounds
    bounds_dict = {}

    # get a list of all columns that are not object type
    non_object_cols = df.dtypes[df.dtypes != 'object'].index


    for col in non_object_cols:
        # get necessary iqr values
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr

        #store values in a dictionary referencable by the column name
        #and specific bound
        bounds_dict[col] = {}
        bounds_dict[col]['upper_bound'] = upper_bound
        bounds_dict[col]['lower_bound'] = lower_bound

    for col in non_object_cols:
        #retrieve bounds
        col_upper_bound = bounds_dict[col]['upper_bound']
        col_lower_bound = bounds_dict[col]['lower_bound']

        #remove rows with an outlier in that column
        df = df[(df[col] < col_upper_bound) & (df[col] > col_lower_bound)]
    
    return df

#Encode categorical columns using a one hot encoder (pd.get_dummies).
def encode_gender(df):
    df['is_female'] = df.gender == 'Female'
    df = df.drop(columns='gender')
    return df

# Split data into train, validate, and test.
def split_data(df):
    '''
    This function performs split on mall data.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123) 
    
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123) 
    return train, validate, test
  
# Summarize the data (include distributions and descriptive statistics).
def overview_describe(df):
    print('--- Shape: {}'.format(df.shape))
    print()
    print('--- Info')
    df.info()
    print()
    print('--- Column Descriptions')
    print(df.describe(include='all'))  

# Scaling
def scaling_mall_df(train, validate, test, columns_to_scale):

    '''
    This function takes in a data set that is split , makes a copy and uses the min max scaler to scale all three data sets. additionally it adds the columns names on the scaled data and returns trainedscaled data, validate scaled data and test scale
    '''
   #columns to scale
    columns_to_scale = ['age', 'spending_score', 'annual_income']
    #copying the dataframes for distinguishing between scaled and unscaled data
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    # defining the minmax scaler 
    scaler = MinMaxScaler()
    
    #scaling the trained data and giving the scaled data column names 
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.fit_transform(train[columns_to_scale]), 
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
    
    #scaling the validate data and giving the scaled data column names 
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    
    #scaling the test data and giving the scaled data column names 
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])

    #returns three dataframes; train_scaled, validate_scaled, test_scaled
    return train_scaled, validate_scaled, test_scaled
