import pandas as pd
import numpy as np
import json

def min_max_scaling(data):
    min_values = data.min()
    max_values = data.max()

    scaled_data = (data - min_values) / (max_values - min_values)

    ranges = {
        'min_values': min_values.to_dict(),
        'max_values': max_values.to_dict()
    }

    with open('data/scaling_params.json', 'w') as f:
        json.dump(ranges, f)

    return scaled_data

df = pd.read_csv('data/sample_data.csv')

features_df = df.drop('good_partner', axis=1)
scaled_features = min_max_scaling(features_df)

scaled_df = pd.DataFrame(scaled_features)
scaled_df['good_partner'] = df['good_partner']

scaled_df.to_csv('data/sample_data.csv', index=False)