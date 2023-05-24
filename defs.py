# import pandas as pd
from sklearn.metrics import *
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import StratifiedKFold
# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import warnings
# import pickle
from sklearn.base import clone
# from sklearn.compose import make_column_transformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from sklearn.model_selection import (
#     cross_val_score, StratifiedKFold,
#     train_test_split, HalvingGridSearchCV
# )
from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
# from sklearn.svm import LinearSVC
# from sklearn.tree import DecisionTreeClassifier



def label_carrier_train(carrier):
    if carrier in ['WN','AA', 'DL', 'OO', 'UA']:
        return 3
    if carrier in ['YX', '9E', 'MQ', 'B6', 'OH']:
        return 2
    if carrier in ['AS', 'NK', 'F9', 'G4', 'YV', 'QX', 'HA']:
        return 1



def process_features(final_df):
    final_df = final_df.copy()
    # Create is_weekend col
    final_df['is_weekend'] = final_df['DAY_OF_WEEK'].isin([1, 7]).astype(int)

    # Drop day and time columns
    final_df.drop(['DAY_OF_WEEK'], axis=1, inplace=True)

    #Bin and OHE Carriers
    final_df['carrier_popularity'] = final_df['OP_CARRIER'].apply(lambda x: label_carrier_train(x))

#     # Get avg delay
#     final_df = add_mean_column(final_df, 'origin_avg_time.csv', 'ORIGIN')
#     final_df = add_mean_column(final_df, 'dest_avg_time.csv', 'DEST')
#     final_df = add_mean_column(final_df, 'tailnumber_avg_time.csv', 'TAIL_NUM')
    
    le = LabelEncoder()
    le.fit(final_df['OP_CARRIER'])
    final_df['OP_CARRIER'] = le.transform(final_df['OP_CARRIER'])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("OP_CARRIER_mapping", mapping)
    
    le = LabelEncoder()
    le.fit(final_df['ORIGIN'])
    final_df['ORIGIN'] = le.transform(final_df['ORIGIN'])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("ORIGIN_mapping", mapping)
    
    le = LabelEncoder()
    le.fit(final_df['DEST'])
    final_df['DEST'] = le.transform(final_df['DEST'])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("DEST_mapping", mapping)
#     final_df = final_df[((final_df['ARR_TIME']<=1440) & (final_df['DEP_TIME']<=1440))]
    
    # Impute nulls with 0
    final_df= final_df.fillna(0)

    return final_df
     

def create_pipeline(classifier):
    pipe = make_pipeline(
        # clone(transform_features),
        MinMaxScaler(),
        classifier
    )
    return pipe


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
