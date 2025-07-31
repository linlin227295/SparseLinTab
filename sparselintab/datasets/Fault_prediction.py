import numpy as np
import pandas as pd

from sparselintab.datasets.base import BaseDataset


class FaultPredictionDataSet(BaseDataset):
    def __init__(self, c):
        super().__init__(fixed_test_set_index=None)
        self.c = c

    def load(self):
        """
        加载 使用Sensor数据进行故障预测 数据集
        """
        # 此处可以加入数据的预处理
        self.data_table = pd.read_csv('data/fault-prediction/data.csv', header=None).to_numpy()
        self.data_table = self.data_table[1:, :]
        self.N = self.data_table.shape[0]
        self.D = self.data_table.shape[1]
        self.num_target_cols = []
        self.cat_target_cols = [self.D - 1]
        self.cat_features = [self.D - 1]
        self.num_features = list(range(0, self.D - 1))
        self.missing_matrix = np.zeros((self.N, self.D), dtype=np.bool_)
        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['fault_pre']


if __name__ == '__main__':
    test = FaultPredictionDataSet(1)
    test.load()
    print("data_table:", test.data_table)
    print("N:", test.N)
    print("D", test.D)
    print("num_target_cols:", test.num_target_cols)
    print("cat_target_cols:", test.cat_target_cols)
    print("cat_features:", test.cat_features)
    print("num_features:", test.num_features)
    print("missing_matrix:", test.missing_matrix)
    print("is_data_loaded:", test.is_data_loaded)
    print("tmp_file_or_dir_names:", test.tmp_file_or_dir_names)
