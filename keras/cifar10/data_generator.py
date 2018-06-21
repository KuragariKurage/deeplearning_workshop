import numpy as np
import cv2
from keras.utils import np_utils
from util import array_operation as my_array_operation

class dataGenerator (object):

    def __init__(self, dataset_file, num_output, batch_size=32, shuffle=True):
        self.data_path_list = []
        self.label_list = []
        self.batch_size = batch_size
        self.shuffle = shuffle

        for line in open(dataset_file, "r"):
            data_path, label = line.split(",")

            self.data_path_list.append(data_path)
            self.label_list.append(int(label))

        self.label_list = np_utils.to_categorical(self.label_list, num_output)

    # 学習用のバッチを取り出す関数(ジェネレータ)
    def get_batch(self):
        while 1:
            if self.shuffle is True:
                self.data_path_list, self.label_list = my_array_operation.shuffle_two_list(self.data_path_list, self.label_list)

            for i in range(0, len(self.data_path_list), self.batch_size):
                # データをロード
                data = []
                for data_path in self.data_path_list[i:i + self.batch_size]:
                    data.append(cv2.imread(data_path, cv2.IMREAD_COLOR))
                label = self.label_list[i:i + self.batch_size]

                # 正規化
                data = np.array(data, dtype=np.float)
                data /= 255

                yield data, np.array(label)

    # 1epochあたりのiteration数を取り出す関数
    def get_iter_num(self):
        return int(len(self.data_path_list) / self.batch_size)

    def get_batch_shape(self):
        batch = []
        for data_path in self.data_path_list[0:0 + self.batch_size]:
            batch.append(cv2.imread(data_path, cv2.IMREAD_COLOR))

        return np.array(batch).shape

def test():
    pathDataDir = "/mnt/data3/nakano/magic/optical_flow/teller_optflow_Farneback_320x180"
    pathBlinkDir = "/mnt/data3/nakano/magic/magic_data/bk_prob.mat"
    trainRate = 0.8

    DG = dataGenerator(pathDataDir, trainRate, pathBlinkDir, valRate=0)
    iterNum = DG.get_train_iter_num()

    testDataNum = 0
    # for testBatch in DG.get_test_batch():
    #     data, target = testBatch
    #     testDataNum += len(data)

    for trainBatch in DG.get_train_batch():
        data, target = trainBatch
        data = np.array(data)
        target = np.array(target)

    print("")

if __name__ == '__main__':
    test()

