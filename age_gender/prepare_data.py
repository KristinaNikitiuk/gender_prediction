import pandas as pd


class PrepareData(object):
    dataframe = pd.read_csv("gmobi_filtered.csv", sep=',')
    result_data = pd.read_csv("gmobi_filtered_results.csv", sep=',')

    def __init__(self):
            pass

    def _get_unpopular_apps(self):
        PrepareData.dataframe['frequency'] = PrepareData.dataframe['app_package'].map(PrepareData.dataframe['app_package'].value_counts())
        unpopular_apps = PrepareData.dataframe.loc[PrepareData.dataframe['frequency'] <= 100].iloc[:, 1]
        app_list = []
        for app in unpopular_apps:
            app_list.append(app)
        return app_list

    def _delete_unpopular_apps(self):
        filtered_df = PrepareData.dataframe[~PrepareData.dataframe['app_package'].isin(self._get_unpopular_apps())]
        return filtered_df.drop(['frequency'], 1)

    def _reformat_df_before_encoding(self):
        reformatted_df = self._delete_unpopular_apps().groupby("tuid")['app_package'].apply(lambda tags: ' '.join(tags)).reset_index(name='app_package')
        return reformatted_df

    def _encode_apps_n_merge_result(self):
        reformatted_df = self._reformat_df_before_encoding()
        encode_app = reformatted_df['app_package'].str.replace(' ', ',').str.get_dummies(sep=',')
        encoded_apps_df = pd.concat([reformatted_df['tuid'], encode_app], axis=1)
        merged_df = pd.merge(encoded_apps_df, PrepareData.result_data[['tuid', 'sex']], on='tuid', how='right')
        return merged_df

    def normalize_data(self):
        prepared_data_df = self._encode_apps_n_merge_result()
        girls_df = prepared_data_df.loc[prepared_data_df['sex'] == 'f']
        boys_df = prepared_data_df.loc[prepared_data_df['sex'] == 'm']
        normalized_df = pd.merge(girls_df, boys_df.head(girls_df.shape[0]), how='outer')
        return normalized_df.dropna()
