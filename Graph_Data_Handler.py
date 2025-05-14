import json
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
from os.path import join
from math import radians, cos, sin, sqrt, atan2
import numpy as np


class FeatureEngineering:

    def __init__(self, use_validation, feature_cols, training_end_date, validation_end_date, test_end_date):
        
        self.input_path = join("io", "input")
        self.output_path = join("io", "output")
        self.edges_df = pd.read_csv(join(self.output_path, "interm_network_edges.csv"), encoding='utf-8', sep=',')
        self.traffic_data_all_merged = pd.read_csv(join(self.output_path, "fct_traffic_data_all_merged.csv"), encoding='utf-8', sep=',')
        self.poi_df = pd.read_csv(join(self.input_path, "base_point_of_interest.csv"), encoding='utf-8', sep=',', index_col=0)
        self.use_validation = use_validation
        self.training_end_date = training_end_date
        self.validation_end_date = validation_end_date
        self.test_end_date = test_end_date
        self.feature_cols = feature_cols
        

    def create_features(self):
        df = self.traffic_data_all_merged
        df['ETA'] = df['ETA'].round().astype(int)
        # Ensure timestamp is a datetime object
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by=['node_id', 'timestamp'], inplace=True)
        # Extracting date components
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['dayofmonth'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['weekofyear'] = df['timestamp'].dt.isocalendar().week
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        # Calculate quarter of the hour (0, 15, 30, 45)
        df['quarter_hour'] = (df['minute'] // 15) * 15
    
        # Trigonometric features for capturing time cycles
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_dayofmonth'] = np.sin(2 * np.pi * df['dayofmonth'] / 31)
        df['cos_dayofmonth'] = np.cos(2 * np.pi * df['dayofmonth'] / 31)
        df['sin_year'] = np.sin(2 * np.pi * df['year'] / df['year'].max())
        df['cos_year'] = np.cos(2 * np.pi * df['year'] / df['year'].max())
        df['sin_weekofyear'] = np.sin(2 * np.pi * df['weekofyear'] / 53)
        df['cos_weekofyear'] = np.cos(2 * np.pi * df['weekofyear'] / 53)
        df['sin_quarter_hour'] = np.sin(2 * np.pi * df['quarter_hour'] / 60)  # 60 minutes in an hour
        df['cos_quarter_hour'] = np.cos(2 * np.pi * df['quarter_hour'] / 60)
    
        # Rolling averages for ETA
        window_sizes = [4, 12, 68, 476, 20240]  # Assuming hourly data
        for window in window_sizes:
            rolling_avg_col = f'rolling_avg_{window}h'
            df[rolling_avg_col] = df.groupby('node_id')['ETA'].transform(lambda x: x.rolling(window=window, closed='left').mean())
    
        # Median ETA for filling NAs
        median_speed = df.groupby('node_id')['ETA'].transform('median')
        
        # Lags
        for lag in [1, 4, 476, 20240]:  # Assuming daily data, adjust if data is hourly
            lag_col = f'lag{lag}h'
            df[lag_col] = df.groupby('node_id')['ETA'].shift(lag)
            df.loc[:, lag_col] = df[lag_col].fillna(median_speed)
    
        # Date indicators
        df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        df['end_of_month'] = ((df['timestamp'] + pd.offsets.MonthEnd(0)) - df['timestamp']).dt.days < 2
        df['end_of_month'] = df['end_of_month'].astype(int)
        df['start_of_month'] = (df['timestamp'] - (df['timestamp'] - pd.offsets.MonthBegin(1))).dt.days < 2
        df['start_of_month'] = df['start_of_month'].astype(int)

    
        # Fill rolling averages where NaN with median_speed
        for window in window_sizes:
            rolling_avg_col = f'rolling_avg_{window}h'
            df.loc[:, rolling_avg_col] = df[rolling_avg_col].fillna(median_speed)

        df['ETA_curr'] = df['ETA']

        df.sort_values(by=['node_id', 'timestamp'], inplace=True)
        df['target'] = df.groupby('node_id')['ETA'].shift(-1)
        df.dropna(subset=['target'], inplace=True)
    
        return df

    
    def haversine(self, lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  # Radius of Earth in kilometers
        return c * r


    def create_poi_distance(self):
        # Create an empty list to hold the data
        data = []
        stops_df = self.edges_df
        poi_df = self.poi_df
        # Iterate over each stop in stops_df
        for idx, stop in stops_df.iterrows():
            # Get the stop details
            stop_id = stop['edge_id']
            stop_lat = stop['start_lat']
            stop_lon = stop['start_long']  # Assuming this column name needs to be corrected in your description
    
            # Calculate distance to each POI and collect the results
            distances = {}
            distances['edge_id'] = stop_id
            
            # Maintain a counter for naming POI columns
            poi_counter = 1
            
            for jdx, poi in poi_df.iterrows():
                poi_name = f"poi_{poi_counter}"
                poi_lat = poi['latitude']
                poi_lon = poi['longitude']
                
                # Calculate Haversine distance
                distance = self.haversine(stop_lat, stop_lon, poi_lat, poi_lon)
                distances[poi_name] = distance
                
                # Increment the POI counter
                poi_counter += 1
    
            # Append the results for this stop to the data list
            data.append(distances)
    
        # Create a DataFrame from the collected data
        result_df = pd.DataFrame(data)
        result_df.rename(columns={'edge_id': 'node_id'}, inplace=True)
        result_df['node_id'] = result_df['node_id'].astype(str)
        columns = result_df.columns[1:].tolist()
        return result_df, columns


    def split_time_series_data(self, df):
        """
        Splits the data into training and evaluation sets based on provided end dates.
        """
        # Convert timestamp column to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
        if self.use_validation:
            # Split data for training and validation
            train_df = df[df['timestamp'] <= pd.to_datetime(self.training_end_date)]
            evaluation_df = df[(df['timestamp'] > pd.to_datetime(self.training_end_date)) & 
                               (df['timestamp'] <= pd.to_datetime(self.validation_end_date))]
        else:
            # Split data for training (including what would have been validation) and test
            train_df = df[df['timestamp'] <= pd.to_datetime(self.validation_end_date)]
            evaluation_df = df[(df['timestamp'] > pd.to_datetime(self.validation_end_date)) & 
                               (df['timestamp'] <= pd.to_datetime(self.test_end_date))]
    
        return train_df, evaluation_df

    
    def scale_df(self, df, cols_to_scale, scaler=None):
        # Create a deep copy of the DataFrame to avoid modifying the original data
        scaled_df = df[cols_to_scale].copy()
    
        if scaler is None:
            scaler = {}
            for col in cols_to_scale:
                mean = scaled_df[col].mean()
                std = scaled_df[col].std()
                # Prevent division by zero by setting std to 1 if it is 0 (no variation)
                if std == 0:
                    std = 1
                scaler[col] = {'mean': mean, 'std': std}
                scaled_df[col] = (scaled_df[col] - mean) / std
    
        else:
            # Apply the scaler passed as parameter
            for col in cols_to_scale:
                mean = scaler[col]['mean']
                std = scaler[col]['std']
                scaled_df[col] = (scaled_df[col] - mean) / std
    
        # Concatenate the non-scaled columns
        final_df = pd.concat([df.drop(cols_to_scale, axis=1), scaled_df], axis=1)
        return final_df, scaler
    

    def get_datasets(self):
        print('Create features')
        traffic_data = self.create_features()
        
        print('Calculate Poi distances')
        poi_distance, poi_columns = self.create_poi_distance()
        traffic_data = pd.merge(traffic_data, poi_distance, on='node_id', how='left')
        
        print('Training-Evaluation set split')
        train_df, eval_df = self.split_time_series_data(traffic_data)
        del traffic_data

        print('Standardization')
        feature_cols_num = [col for col in self.feature_cols if col!='node_id']
        feature_cols_scale = feature_cols_num + poi_columns
        trainset_full, scaller = self.scale_df(train_df, feature_cols_scale)
        del trainset_full
        train_df, scaller1 = self.scale_df(train_df, feature_cols_scale, scaller)
        eval_df, scaller1 = self.scale_df(eval_df, feature_cols_scale, scaller)
        
        return train_df, eval_df, poi_columns