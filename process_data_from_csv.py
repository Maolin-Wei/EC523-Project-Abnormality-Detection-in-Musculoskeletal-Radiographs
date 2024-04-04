import pandas as pd
from collections import defaultdict

def read_data_from_csv():
    train_df = pd.read_csv('MURA-v1.1/train_labeled_studies.csv')
    valid_df = pd.read_csv('MURA-v1.1/valid_labeled_studies.csv')

    def prepare_data(df):
        data = []
        for index, row in df.iterrows():
            image_path = row[0]  # Assuming the first column contains the image paths
            label = row[1]       # Assuming the second column contains the labels
            data.append((image_path, label))
        return data

    train_data = prepare_data(train_df)
    valid_data = prepare_data(valid_df)



    def divide_by_body_part(data):
        body_parts_data = defaultdict(list)
        for path, label in data:
            body_part = path.split('/')[2]  # Assuming the third component is the body part
            body_parts_data[body_part].append((path, label))
        return body_parts_data

    train_data_by_body_part = divide_by_body_part(train_data)
    valid_data_by_body_part = divide_by_body_part(valid_data)

    train_body_parts = {part: len(train_data_by_body_part[part]) for part in train_data_by_body_part}
    valid_body_parts = {part: len(valid_data_by_body_part[part]) for part in valid_data_by_body_part}

    train_data = {}
    valid_data = {}

    def create_dfs_by_body_part(data_by_body_part, dfs_by_body_part):
        for body_part, data in data_by_body_part.items():
            df = pd.DataFrame(data, columns=['Image_Path', 'Label'])
            dfs_by_body_part[body_part] = df

    create_dfs_by_body_part(train_data_by_body_part, train_data)
    create_dfs_by_body_part(valid_data_by_body_part, valid_data)

    return train_data, valid_data
