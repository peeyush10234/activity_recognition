import glob
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def upsample_label(temp_x, temp_y):
    y_time_list = temp_y['time'].values
    y_label = temp_y['label'].values
    jdx = 0
    label_list = []
    for index, row in temp_x.iterrows():
        try:
            if row['time'] > y_time_list[jdx]:
                jdx += 1
            label_list.append(y_label[jdx])
        except:
            label_list.append(0)
    return label_list


class DataPreprocessing:

    def __init__(self, data_dir_path, scaler = None,
                 time_steps=1, step=1, is_train=True,
                 val_size=0.2):

        self.data_dir_path = data_dir_path
        self.scaler = scaler
        self.time_steps = time_steps
        self.step = step
        self.is_train = is_train
        self.val_size = val_size
        self.X = None
        self.Y = None

    def load_data(self):
        column_list = ['acc_x', 'acc_y',
                       'acc_z', 'gyro_x',
                       'gyro_y', 'gyro_z',
                       'subject', 'time', 'label']

        df_data = pd.DataFrame([], columns=column_list)

        for idx in glob.glob(self.dir_path + '*.csv'):
            df_type = idx.split('.')[0].split('__')[1]
            if df_type == 'x':
                subject_name = idx.split('.')[0].split('__')[0].split('/')[-1]
                x_path = idx.split('.')[0].split('__')[0] + '__' + df_type + '.csv'
                x_time_path = idx.split('.')[0].split('__')[0] + '__' + 'x_time' + '.csv'
                y_path = idx.split('.')[0].split('__')[0] + '__' + 'y' + '.csv'
                y_time_path = idx.split('.')[0].split('__')[0] + '__' + 'y_time' + '.csv'

                df_x = pd.read_csv(x_path)

                sub_name = [subject_name] * df_x.shape[0]
                df_x['subject'] = sub_name

                df_x_time = pd.read_csv(x_time_path)
                df_x['time'] = df_x_time

                df_y = pd.read_csv(y_path)
                sub_name = [subject_name] * df_y.shape[0]
                df_y['subject'] = sub_name

                df_y_time = pd.read_csv(y_time_path)
                df_y['time'] = df_y_time
                df_y.columns = ['label', 'subject', 'time']
                label_list = upsample_label(df_x, df_y)
                df_x['label'] = label_list
                df_x.columns = column_list
                df_data = pd.concat([df_data, df_x], ignore_index=True)

        self.df_data = df_data


    def scale_features(self):
        self.scale_columns = self.df_data.columns[:6]
        scaler_obj = self.scaler.fit(self.df_data[self.scale_columns])

        self.df_data.loc[:, self.scale_columns] = scaler_obj.transform(
            self.df_data[self.scale_columns].to_numpy()
        )

    def create_windows(self):
        X = self.df_data[self.scale_columns]
        if self.is_train:
            y = self.df_data.label

        for i in range(0, len(X) - self.time_steps, self.step):
            v = X.iloc[i:(i + self.time_steps)].values
            X.append(v)

            if self.is_train:
                labels = y.iloc[i: i + self.time_steps]
                y.append(self.stats.mode(labels)[0][0])

        if self.is_train:
            return np.array(X), np.array(y).reshape(-1, 1)

        return np.array(X), None

    def under_sampler(self):
        df_label = pd.DataFrame(self.y, columns=['label'])
        df_label['index_'] = df_label.index

        ## Library for performing undersampling
        rus = RandomUnderSampler(sampling_strategy='not minority', random_state=1)
        df_balanced, balanced_labels = rus.fit_resample(df_label, df_label['label'])
        df_balanced = pd.DataFrame(df_balanced, columns=['label', 'index_'])

        self.X = self.X[df_balanced['index_'].values]
        self.y = self.y[df_balanced['index_'].values]

    def one_hot_encoding(self):
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        enc = enc.fit(self.y_train)
        y_train = enc.transform(self.y_train)
        y_val = enc.transform(self.y_val)


    def perform_pre_processing(self):
        self.load_data()
        self.scale_features()
        self.X, self.Y = self.create_windows()
        self.X = np.expand_dims(self.X, axis=-1)
        self.under_sampler()

        val_size = self.val_size  # validation data size
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=val_size)

        self.one_hot_encoding()

        return self.X_train, self.y_train, self.X_val, self.y_val





