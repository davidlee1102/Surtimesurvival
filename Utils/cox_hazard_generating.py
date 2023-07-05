import os
import math
import numpy as np
import pandas as pd

from calculation_function import finding_time_t


def input_features(features):
    z_max = []
    z_min = []
    features_len = len(features)
    for i in features:
        if (type(i) is not list) or (type(i[0]) is str) or (type(i[1]) is str):
            print("Data error, please check again")
        else:
            if i[0] < i[1]:
                z_min.append(i[0])
                z_max.append(i[1])
            else:
                print(f"the value of min and max have contrasted at {i}, we have applied again for this")
                z_min.append(i[1])
                z_max.append(i[0])

    features_df = pd.DataFrame({'Z_Min': z_min, 'Z_Max': z_max})
    return features_len, features_df


def input_dataframe(path, cat_feat: list = [], num_feat: list = []):
    assert os.path.exists(path), "File not found, please check"
    df = pd.read_csv(path)
    try:
        time_event_df = df[['event', 'time']]
        df = df.drop(columns=['event', 'time'])
    except Exception as E:
        print(E)
    min_max_df = df.agg(['min', 'max'])
    features_len = len(list(df.columns))
    min_max_df = min_max_df.transpose()
    min_max_df = min_max_df.rename(columns={"min": "Z_Min", "max": "Z_Max"})
    return time_event_df, min_max_df, features_len


def generate_z(features_df, n_generate):
    results = pd.DataFrame()
    for index, row in features_df.iterrows():
        min_val, max_val = row['Z_Min'], row['Z_Max']
        if min_val == 0 and max_val == 1:
            samples = np.random.choice([0, 1], n_generate)
        else:
            samples = np.random.uniform(min_val, max_val, n_generate)
        results = pd.concat([results, pd.DataFrame([samples])])
    results = results.reset_index(drop=True)

    return results


def generate_u(n_generate):
    samples = np.random.uniform(0, 1, n_generate)
    samples = pd.DataFrame(samples)
    samples = samples.transpose()
    return samples


def input_beta(beta_list: list = [], features_len: int = 0):
    assert len(beta_list) == features_len, "the length of features is not equal, please check"
    return beta_list


def time_failure_calculation(n_generate, features, path, k, g, beta_list):
    if len(beta_list) == 0 or (len(features) == 0 and path is None):
        print("Error in data input, please check")
        return None
    if len(features) == 0:
        time_event_df, min_max_df, features_len = input_dataframe(path)
        z_generated = generate_z(min_max_df, n_generate)
    else:
        features_len, features_df = input_features(features)
        z_generated = generate_z(features_df, n_generate)

    beta_list = input_beta(beta_list, features_len)
    beta_list = np.array(beta_list)
    z_generated = np.array(z_generated)
    list_harzard = np.dot(z_generated.T, beta_list)
    u = generate_u(n_generate)

    list_failure_time = []
    for i in range(n_generate):
        result = finding_time_t(u[i], k, g, list_harzard[i])
        list_failure_time.append(result)

    return list_failure_time
