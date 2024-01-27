import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# parsing arch_str
def parser_arch_str(arch_str):
    sub_connection = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
    splited_str = re.findall(r'(\w+)~\d+\|', arch_str)
    node_con_index = []
    for i in range(len(splited_str)):
        node_con_index.append(sub_connection.index(splited_str[i]))
    return node_con_index
            

# encode architecture to one-hot code
def encode_arch(file_path):
    arch_data = pd.read_excel(file_path, sheet_name="Sheet1",  usecols=["arch_str"], header=0)
    arch_data = pd.concat([arch_data, pd.DataFrame(columns=["node1~0", "node2~0", "node2~1", "node3~0", "node3~1", "node3~2"])], sort=False)
    for i in range(arch_data.shape[0]):
        arch_str = arch_data["arch_str"][i]
        node_con_index = parser_arch_str(arch_str)
        arch_data["node1~0"][i] = node_con_index[0]
        arch_data["node2~0"][i] = node_con_index[1]
        arch_data["node2~1"][i] = node_con_index[2]
        arch_data["node3~0"][i] = node_con_index[3]
        arch_data["node3~1"][i] = node_con_index[4]
        arch_data["node3~2"][i] = node_con_index[5]

    ohe_column = pd.get_dummies(arch_data, columns=["node1~0", "node2~0", "node2~1", "node3~0", "node3~1", "node3~2"])
    return ohe_column


# get features of all dataset
def get_features(file_path):
    data = pd.read_excel(file_path, sheet_name="Sheet1",  usecols=["flops", "input_size"], header=0)
    encoded_arch = encode_arch(file_path)
    features = pd.concat([data, encoded_arch], axis=1, sort=False)
    features = features.drop(["arch_str"], axis=1)
    features = torch.tensor(features[:features.shape[0]].values, dtype=torch.float)
    return features


# get input features from arch_str
def get_features_by_arch_str(arch_str, flops, input_size):
    features = [flops, input_size]
    encoded_arch = parser_arch_str(arch_str)
    one_hot_arch_code = [0] * 30
    features = [input_size, flops]
    for i in range(len(encoded_arch)):
        if encoded_arch[i] == 0:
            one_hot_arch_code[5*i] = 1
            one_hot_arch_code[5*i+1] = 0
            one_hot_arch_code[5*i+2] = 0
            one_hot_arch_code[5*i+3] = 0
            one_hot_arch_code[5*i+4] = 0
        elif encoded_arch[i] == 1:
            one_hot_arch_code[5*i] = 0
            one_hot_arch_code[5*i+1] = 1
            one_hot_arch_code[5*i+2] = 0
            one_hot_arch_code[5*i+3] = 0
            one_hot_arch_code[5*i+4] = 0
        elif encoded_arch[i] == 2:
            one_hot_arch_code[5*i] = 0
            one_hot_arch_code[5*i+1] = 0
            one_hot_arch_code[5*i+2] = 1
            one_hot_arch_code[5*i+3] = 0
            one_hot_arch_code[5*i+4] = 0
        elif encoded_arch[i] == 3:
            one_hot_arch_code[5*i] = 0
            one_hot_arch_code[5*i+1] = 0
            one_hot_arch_code[5*i+2] = 0
            one_hot_arch_code[5*i+3] = 1
            one_hot_arch_code[5*i+4] = 0
        elif encoded_arch[i] == 4:
            one_hot_arch_code[5*i] = 0
            one_hot_arch_code[5*i+1] = 0
            one_hot_arch_code[5*i+2] = 0
            one_hot_arch_code[5*i+3] = 0
            one_hot_arch_code[5*i+4] = 1
    features = features + one_hot_arch_code
    features = torch.tensor(features, dtype=torch.float)
    return features
    

# get dataset lables
def get_labels(file_path):
    labels = pd.read_excel(file_path, sheet_name="Sheet1",  usecols=["nezhaD1"], header=0)
    labels = torch.tensor(labels.values, dtype=torch.float).view(-1, 1)
    return labels


# get all the dataset
def get_dataset(file_path):
    data = pd.read_excel(file_path, sheet_name="Sheet1",  usecols=["flops", "input_size"], header=0)
    labels = pd.read_excel(file_path, sheet_name="Sheet1",  usecols=["nezhaD1"], header=0)
    encoded_arch = encode_arch(file_path)
    dataset = pd.concat([data, encoded_arch, labels], axis=1, sort=False)
    dataset = dataset.drop(["arch_str"], axis=1)
    return dataset


class CSVDataset(Dataset):
    def __init__(self, data_path_list):
        df = get_dataset(data_path_list)
        self.x = df.values[:, :-1]
        self.y = df.values[:, -1]

        self.x = self.x.astype("float32")
        self.y = self.y.astype("float32")
        self.y = self.y.reshape((-1, 1))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return [self.x[index], self.y[index]]
    
    def get_splits(self, n_test=0.3):
        test_size = round(n_test * len(self.x))
        train_size = len(self.x) - test_size
        return random_split(self, [train_size, test_size])

    def prepare_data(path):
        dataset = CSVDataset(path)
        train, test = dataset.get_splits()
        train_dl = DataLoader(train, batch_size=1024, shuffle=True)
        test_dl = DataLoader(test, batch_size=1024, shuffle=False)
        return train_dl, test_dl