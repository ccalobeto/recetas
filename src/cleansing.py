import json
import pandas as pd
import numpy as np


""" DESIGN SCHEMA """
## First commit

# LOCAL FUNCTIONS
# GLOBAL VARIABLES
# IMPORT JSON FILE
# CLEAN JSON FILE
# PREPARE JDATAFRAME
## convert list of dicts to dataframe
# EXPORT DATA

# GLOBAL VARIABLES
INPUT_PATH = '../examples/'
OUTPUT_PATH = '../output/'
FILE_1_ = 'train.json'
INGREDIENT_THRESHOLD = .1

# IMPORT JSON FILE
with open(INPUT_PATH + FILE_1_, 'r') as file:
    jsonList = json.load(file)

# PREPARE DATAFRAME
## convert list of dicts to dataframe
df_ = pd.DataFrame(jsonList)

## clear dataset

## expand dataframe with list of values
lst_col = 'ingredients'
df = pd.DataFrame({col: np.repeat(df_[col].values, df_[lst_col].str.len())
                   for col in df_.columns.drop(lst_col)}
                  ).assign(**{lst_col: np.concatenate(df_[lst_col].values)})[df_.columns]
df.rename(columns={'ingredients': 'ingredient'}, inplace=True)

## aggregations
aggcuisine_df = df_.groupby('cuisine')['ingredients'].count().to_frame().rename(columns={'ingredients': 'cuisineCount'})

aggingredients_df = df.groupby(['cuisine', 'ingredient'])['id'].count().\
    to_frame().rename(columns={'id': 'ingredientCount'})

cuisine_df = aggingredients_df.join(aggcuisine_df).reset_index()

## adding some features
cuisine_df['ingredientRatio'] = cuisine_df['ingredientCount'] / cuisine_df['cuisineCount']
cuisine_df['isAboveThreshold'] = cuisine_df['ingredientRatio'].apply(lambda x: True if x > .1 else False)
cuisine_df['totalIngredientCount'] = cuisine_df.groupby('ingredient')['ingredientCount'].transform('sum')
cuisine_df['relativeUsage'] = cuisine_df['ingredientRatio'] / (cuisine_df['totalIngredientCount'] /
                                                               aggcuisine_df['cuisineCount'].sum())
