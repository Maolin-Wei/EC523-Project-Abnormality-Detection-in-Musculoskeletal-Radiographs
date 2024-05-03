import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

def read_data_from_csv():
    train_df = pd.read_csv('MURA-v1.1/train_labeled_studies.csv')
    valid_df = pd.read_csv('MURA-v1.1/valid_labeled_studies.csv')

    def prepare_data(df):
        data = []
        for index, row in df.iterrows():
            image_path = row[0] 
            label = row[1]
            data.append((image_path, label))
        return data

    train_data = prepare_data(train_df)
    valid_data = prepare_data(valid_df)

    def divide_by_body_part(data):
        body_parts_data = defaultdict(list)
        for path, label in data:
            body_part = path.split('/')[2]
            body_parts_data[body_part].append((path, label))
        return body_parts_data

    train_data_by_body_part = divide_by_body_part(train_data)
    valid_data_by_body_part = divide_by_body_part(valid_data)

    new_valid_data_by_body_part = defaultdict(list)
    test_data_by_body_part = defaultdict(list)

    # Stratified division of valid data into new valid and test datasets
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for body_part, data in valid_data_by_body_part.items():
        paths, labels = zip(*data)
        df = pd.DataFrame(data, columns=['Image_Path', 'Label'])
        for train_index, test_index in splitter.split(paths, labels):
            new_valid_data_by_body_part[body_part] = [data[i] for i in train_index]
            test_data_by_body_part[body_part] = [data[i] for i in test_index]

    train_data = {}
    valid_data = {}
    test_data = {}

    def create_dfs_by_body_part(data_by_body_part, dfs_by_body_part):
        for body_part, data in data_by_body_part.items():
            df = pd.DataFrame(data, columns=['Image_Path', 'Label'])
            dfs_by_body_part[body_part] = df

    create_dfs_by_body_part(train_data_by_body_part, train_data)
    create_dfs_by_body_part(new_valid_data_by_body_part, valid_data)
    create_dfs_by_body_part(test_data_by_body_part, test_data)

    return train_data, valid_data, test_data

train_data, valid_data, test_data = read_data_from_csv()
