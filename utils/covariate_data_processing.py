import torch
import pandas as pd

from SurvTRACE.survtrace.utils import LabelTransform
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler


# Check sequence length
def padded_mask_processing(df_train):
    max_seq_length = df_train.groupby("seq_id").size().max()
    num_patients = len(df_train["seq_id"].unique())
    print(max_seq_length, max_seq_length)
    padded_patients = []
    masks = []
    for patient_id, patient_data in df_train.groupby("seq_id"):
        padding_rows = max_seq_length - len(patient_data)

        current_patients = torch.zeros(max_seq_length, df_train.shape[1])
        curent_masks = torch.zeros(max_seq_length)
        current_patients[:len(patient_data)] = torch.tensor(patient_data.to_numpy())
        curent_masks[:len(patient_data)] = 1
        masks.append(curent_masks)
        padded_patients.append(current_patients)
    padded_patients = torch.stack(padded_patients)
    masks = torch.stack(masks)
    padded_patients = padded_patients[:, :, 1:]
    return masks, padded_patients


import numpy as np
from SurvTRACE.survtrace.config import STConfig


def pbc2_proccess_covariate(df):
    get_target = lambda df: (df['duration'].values, df['event'].values)
    horizons = [.25, .5, .75]
    times = np.quantile(df["duration"][df["event"] == 1.0], horizons).tolist()

    # cols_standardize = ["seq_temporal_SGOT", "seq_temporal_age", "seq_temporal_albumin", "seq_temporal_alkaline", "seq_temporal_platelets", "seq_temporal_prothrombin", "seq_temporal_serBilir", "seq_temporal_serChol"]
    # cols_categorical = ["seq_static_sex", "seq_temporal_ascites", "seq_temporal_drug", "seq_temporal_edema", "seq_temporal_hepatomegaly", "seq_temporal_histologic", "seq_temporal_spiders"]

    cols_standardize = ["seq_temporal_SGOT", "seq_temporal_age", "seq_temporal_albumin", "seq_temporal_alkaline",
                        "seq_temporal_platelets", "seq_temporal_prothrombin", "seq_temporal_serBilir",
                        "seq_temporal_serChol"]
    cols_categorical = ['seq_static_sex_1.0', 'seq_temporal_ascites_1.0', 'seq_temporal_ascites_2.0',
                        'seq_temporal_drug_1.0', 'seq_temporal_edema_1.0', 'seq_temporal_edema_2.0',
                        'seq_temporal_hepatomegaly_1.0', 'seq_temporal_hepatomegaly_2.0', 'seq_temporal_histologic_1.0',
                        'seq_temporal_histologic_2.0', 'seq_temporal_histologic_3.0', 'seq_temporal_spiders_1.0',
                        'seq_temporal_spiders_2.0']

    df_feat = df.drop(["duration", "event"], axis=1)
    df_feat_standardize = df_feat[cols_standardize]
    df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
    df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

    # must be categorical feature ahead of numerical features!
    df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)

    vocab_size = 0
    for _, feat in enumerate(cols_categorical):
        df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
        vocab_size = df_feat[feat].max() + 1
    print(vocab_size)

    # get the largest duraiton time
    max_duration_idx = df["duration"].argmax()

    # # Hung edited here
    df_train = df_feat.iloc[0:]  #

    # assign cuts
    labtrans = LabelTransform(cuts=np.array([df["duration"].min()] + times + [df["duration"].max()]))
    labtrans.fit(*get_target(df.loc[df_train.index]))
    y = labtrans.transform(*get_target(df))  # y = (discrete duration, event indicator)
    df_y_train = pd.DataFrame(
        {"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion": y[2][df_train.index]},
        index=df_train.index)
    STConfig['labtrans'] = labtrans
    STConfig['num_numerical_feature'] = int(len(cols_standardize))
    STConfig['num_categorical_feature'] = int(len(cols_categorical))
    STConfig['num_feature'] = int(len(df_train.columns))
    STConfig['vocab_size'] = int(vocab_size)
    STConfig['duration_index'] = labtrans.cuts
    STConfig['out_feature'] = int(labtrans.out_features)
    print("______")
    print(int(len(cols_standardize)))
    print("______")
    print(int(len(cols_categorical)))
    print("______")
    print(int(len(df_train.columns)))
    print("______")
    print(int(vocab_size))
    print("______")
    print(labtrans.cuts)
    print("______")
    print(int(labtrans.out_features))

    return y, df, df_train, df_y_train


def pbc2_proccess_covariate_firstsolution(df, location_test):
    get_target = lambda df: (df['duration'].values, df['event'].values)
    horizons = [.25, .5, .75]
    times = np.quantile(df["duration"][df["event"] == 1.0], horizons).tolist()

    # cols_standardize = ["seq_temporal_SGOT", "seq_temporal_age", "seq_temporal_albumin", "seq_temporal_alkaline", "seq_temporal_platelets", "seq_temporal_prothrombin", "seq_temporal_serBilir", "seq_temporal_serChol"]
    # cols_categorical = ["seq_static_sex", "seq_temporal_ascites", "seq_temporal_drug", "seq_temporal_edema", "seq_temporal_hepatomegaly", "seq_temporal_histologic", "seq_temporal_spiders"]

    # cols_standardize = ["seq_temporal_SGOT", "seq_temporal_age", "seq_temporal_albumin", "seq_temporal_alkaline", "seq_temporal_platelets", "seq_temporal_prothrombin", "seq_temporal_serBilir", "seq_temporal_serChol"]
    # cols_categorical = ['seq_static_sex_1.0', 'seq_temporal_ascites_1.0', 'seq_temporal_ascites_2.0', 'seq_temporal_drug_1.0', 'seq_temporal_edema_1.0', 'seq_temporal_edema_2.0', 'seq_temporal_hepatomegaly_1.0', 'seq_temporal_hepatomegaly_2.0', 'seq_temporal_histologic_1.0', 'seq_temporal_histologic_2.0', 'seq_temporal_histologic_3.0', 'seq_temporal_spiders_1.0', 'seq_temporal_spiders_2.0']

    cols_standardize = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                        '17', '18', '19', '20']
    cols_categorical = []

    df_feat = df.drop(["duration", "event"], axis=1)
    df_feat_standardize = df_feat[cols_standardize]
    df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
    df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

    # must be categorical feature ahead of numerical features!
    try:
        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)
    except:
        print("Error Here")

    vocab_size = 0
    for _, feat in enumerate(cols_categorical):
        df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
        vocab_size = df_feat[feat].max() + 1
    print(vocab_size)

    # get the largest duraiton time
    max_duration_idx = df["duration"].argmax()
    # df_test = df_feat.drop(max_duration_idx).sample(frac=0.2)
    # df_train = df_feat.drop(df_test.index)
    # df_val = df_train.drop(max_duration_idx).sample(frac=0.2)
    # df_train = df_train.drop(df_val.index)

    # # # Hung edited here
    # df_train = df_feat.iloc[125:]  #
    # df_train = df_train.sample(frac=1)
    #
    # df_remaining = df_feat.iloc[:124]  #
    # df_remaining = df_remaining.sample(frac=1)
    #
    # # Now let's say you want to split the remaining data into test and validation sets
    # df_test = df_remaining.sample(frac=0.5)  # 20% of the training data for testing
    # df_val = df_remaining.drop(df_test.index)  # 20% of the training data for validating

    # Third revised for prof

    df_test = df_feat.iloc[:location_test]
    df_test = df_test.sample(frac=1)

    location_test = location_test + 1
    df_remaining = df_feat.iloc[location_test:]
    df_remaining = df_remaining.sample(frac=1)

    df_train = df_remaining.iloc[125:]
    df_train = df_train.sample(frac=1)

    df_val = df_feat.iloc[location_test:124]
    df_val = df_val.sample(frac=1)

    # assign cuts
    labtrans = LabelTransform(cuts=np.array([df["duration"].min()] + times + [df["duration"].max()]))
    labtrans.fit(*get_target(df.loc[df_train.index]))
    y = labtrans.transform(*get_target(df))  # y = (discrete duration, event indicator)
    df_y_train = pd.DataFrame(
        {"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion": y[2][df_train.index]},
        index=df_train.index)
    df_y_val = pd.DataFrame(
        {"duration": y[0][df_val.index], "event": y[1][df_val.index], "proportion": y[2][df_val.index]},
        index=df_val.index)
    # df_y_test = pd.DataFrame({"duration": y[0][df_test.index], "event": y[1][df_test.index],  "proportion": y[2][df_test.index]}, index=df_test.index)
    df_y_test = pd.DataFrame({"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})
    STConfig['labtrans'] = labtrans
    STConfig['num_numerical_feature'] = int(len(cols_standardize))
    STConfig['num_categorical_feature'] = int(len(cols_categorical))
    STConfig['num_feature'] = int(len(df_train.columns))
    STConfig['vocab_size'] = int(vocab_size)
    STConfig['duration_index'] = labtrans.cuts
    STConfig['out_feature'] = int(labtrans.out_features)
    print("______")
    print(int(len(cols_standardize)))
    print("______")
    print(int(len(cols_categorical)))
    print("______")
    print(int(len(df_train.columns)))
    print("______")
    print(int(vocab_size))
    print("______")
    print(labtrans.cuts)
    print("______")
    print(int(labtrans.out_features))

    return y, df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val


def pbc2_proccess_covariate_firstsolution_addproportionfirst(df, location_test):
    get_target = lambda df: (df['duration'].values, df['event'].values)
    horizons = [.25, .5, .75]
    times = np.quantile(df["duration"][df["event"] == 1.0], horizons).tolist()

    # cols_standardize = ["seq_temporal_SGOT", "seq_temporal_age", "seq_temporal_albumin", "seq_temporal_alkaline", "seq_temporal_platelets", "seq_temporal_prothrombin", "seq_temporal_serBilir", "seq_temporal_serChol"]
    # cols_categorical = ["seq_static_sex", "seq_temporal_ascites", "seq_temporal_drug", "seq_temporal_edema", "seq_temporal_hepatomegaly", "seq_temporal_histologic", "seq_temporal_spiders"]

    # cols_standardize = ["seq_temporal_SGOT", "seq_temporal_age", "seq_temporal_albumin", "seq_temporal_alkaline", "seq_temporal_platelets", "seq_temporal_prothrombin", "seq_temporal_serBilir", "seq_temporal_serChol"]
    # cols_categorical = ['seq_static_sex_1.0', 'seq_temporal_ascites_1.0', 'seq_temporal_ascites_2.0', 'seq_temporal_drug_1.0', 'seq_temporal_edema_1.0', 'seq_temporal_edema_2.0', 'seq_temporal_hepatomegaly_1.0', 'seq_temporal_hepatomegaly_2.0', 'seq_temporal_histologic_1.0', 'seq_temporal_histologic_2.0', 'seq_temporal_histologic_3.0', 'seq_temporal_spiders_1.0', 'seq_temporal_spiders_2.0']

    cols_standardize = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                        '17', '18', '19', '20', 'proportion_first']
    cols_categorical = []

    df_feat = df.drop(["duration", "event"], axis=1)
    df_feat_standardize = df_feat[cols_standardize]
    df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
    df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

    # must be categorical feature ahead of numerical features!
    try:
        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)
    except:
        print("Error Here")

    vocab_size = 0
    for _, feat in enumerate(cols_categorical):
        df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
        vocab_size = df_feat[feat].max() + 1
    print(vocab_size)

    # get the largest duraiton time
    max_duration_idx = df["duration"].argmax()
    # df_test = df_feat.drop(max_duration_idx).sample(frac=0.2)
    # df_train = df_feat.drop(df_test.index)
    # df_val = df_train.drop(max_duration_idx).sample(frac=0.2)
    # df_train = df_train.drop(df_val.index)

    # # # Hung edited here
    # df_train = df_feat.iloc[125:]  #
    # df_train = df_train.sample(frac=1)
    #
    # df_remaining = df_feat.iloc[:124]  #
    # df_remaining = df_remaining.sample(frac=1)
    #
    # # Now let's say you want to split the remaining data into test and validation sets
    # df_test = df_remaining.sample(frac=0.5)  # 20% of the training data for testing
    # df_val = df_remaining.drop(df_test.index)  # 20% of the training data for validating

    # Third revised for prof

    df_test = df_feat.iloc[:location_test]
    df_test = df_test.sample(frac=1)

    location_test = location_test + 1
    df_remaining = df_feat.iloc[location_test:]
    df_remaining = df_remaining.sample(frac=1)

    df_train = df_remaining.iloc[125:]
    df_train = df_train.sample(frac=1)

    df_val = df_feat.iloc[location_test:124]
    df_val = df_val.sample(frac=1)

    # assign cuts
    labtrans = LabelTransform(cuts=np.array([df["duration"].min()] + times + [df["duration"].max()]))
    labtrans.fit(*get_target(df.loc[df_train.index]))
    y = labtrans.transform(*get_target(df))  # y = (discrete duration, event indicator)
    df_y_train = pd.DataFrame(
        {"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion": y[2][df_train.index]},
        index=df_train.index)
    df_y_val = pd.DataFrame(
        {"duration": y[0][df_val.index], "event": y[1][df_val.index], "proportion": y[2][df_val.index]},
        index=df_val.index)
    # df_y_test = pd.DataFrame({"duration": y[0][df_test.index], "event": y[1][df_test.index],  "proportion": y[2][df_test.index]}, index=df_test.index)
    df_y_test = pd.DataFrame({"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})
    STConfig['labtrans'] = labtrans
    STConfig['num_numerical_feature'] = int(len(cols_standardize))
    STConfig['num_categorical_feature'] = int(len(cols_categorical))
    STConfig['num_feature'] = int(len(df_train.columns))
    STConfig['vocab_size'] = int(vocab_size)
    STConfig['duration_index'] = labtrans.cuts
    STConfig['out_feature'] = int(labtrans.out_features)
    print("______")
    print(int(len(cols_standardize)))
    print("______")
    print(int(len(cols_categorical)))
    print("______")
    print(int(len(df_train.columns)))
    print("______")
    print(int(vocab_size))
    print("______")
    print(labtrans.cuts)
    print("______")
    print(int(labtrans.out_features))

    return y, df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val
