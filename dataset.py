'''
mnist数据集官方下载地址 http://yann.lecun.com/exdb/mnist/

本任务中, 大家只需编辑 DataGenerator 类的 shuffle() 和 normalize() 函数, 其余部分请自行阅读, 以后会用到

为了方便测试, 本任务会检测测试集数据生成器的前100组数据
测试代码如下:

import numpy as np
from dataset import DataGenerator

testGenerator = DataGenerator(mode='test')
testGenerator.load_data()

testGenerator.images = testGenerator.images[:100]
testGenerator.labels = testGenerator.labels[:100]
testGenerator.num_images = 100

testGenerator.shuffle()
testGenerator.normalize()
testIter = testGenerator.getTestIterator()
for images, labels in testIter:
    print(images.mean())
'''

import struct
import numpy as np

class DataGenerator():

    def __init__(self, mode='train', batch_size=1, vaild_ratio=0.2):
        '''
        mode:           该参数有两个选项 'train'和'test', 分别对应训练集的数据生成器和测试集的数据生成器
                        只有mode='train', 参数batch_size和vaild_ratio才是有效的
                        该参数默认为'train'

        batch_size:     当mode='train'时, 由参数batch_size指定训练过程中每次迭代要给模型输入多少个数据样本

        vaild_ratio:    当mode='train'时, 由参数vaild_ratio指定要从全部训练数据中取出多少作为验证集, 注意验证集一般不会参与训练
        '''

        self.mode = mode
        self.batch_size = batch_size
        self.vaild_ratio = vaild_ratio

        self.num_images = 0
        self.raw_images = []
        self.images = []
        self.labels = []

        assert mode in ['train', 'test']

        if mode == 'train':
            self.images_path = '../../mnist/train-images.idx3-ubyte'
            self.labels_path = '../../mnist/train-labels.idx1-ubyte'
        elif mode == 'test':
            self.images_path = '../../mnist/t10k-images.idx3-ubyte'
            self.labels_path = '../../mnist/t10k-labels.idx1-ubyte'

        # if mode == 'train':
        #     self.images_path = '/data/bigfiles/95dee589-911d-4a28-81a3-19af65c83cfe.idx3-ubyte'
        #     self.labels_path = '/data/bigfiles/cc38d1cf-652c-4d10-affd-d9938f79e8a3.idx1-ubyte'
        # elif mode == 'test':
        #     self.images_path = '/data/bigfiles/b8ea2a4b-d5ba-4e53-9d4a-a2b9dc1122b0.idx3-ubyte'
        #     self.labels_path = '/data/bigfiles/e688c6b2-a4ac-41f6-a534-93dbd9bf0409.idx1-ubyte'

    def load_data(self):

        with open(self.images_path, 'rb') as imgpath:
            _, nums, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(nums, rows*cols)

        with open(self.labels_path, 'rb') as lbpath:
            _, nums = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)

        self.num_images = nums
        self.images = [images[i] for i in range(nums)]
        self.labels = [labels[i] for i in range(nums)]

        self.raw_images = np.array(self.images.copy())

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def getTrainIterator(self):
        assert self.mode == 'train'

        train_images = self.images[:int(self.num_images*(1-self.vaild_ratio))]
        train_labels = self.labels[:int(self.num_images*(1-self.vaild_ratio))]

        self.train_nums = train_images.shape[0]
        iter_num = int(np.ceil(self.train_nums / self.batch_size))

        iterator = iter([
                [
                    train_images[i*self.batch_size:(i+1)*self.batch_size], 
                    train_labels[i*self.batch_size:(i+1)*self.batch_size]
                ] for i in range(iter_num)
            ])
        
        return iterator

    def getValidIterator(self):
        assert self.mode == 'train'

        valid_images = self.images[int(self.num_images*(1-self.vaild_ratio)):]
        valid_labels = self.labels[int(self.num_images*(1-self.vaild_ratio)):]

        self.valid_nums = valid_images.shape[0]
        iter_num = self.valid_nums

        iterator = iter([
            [
                valid_images[i],
                valid_labels[i]
            ] for i in range(iter_num)
        ])

        return iterator

    def getTestIterator(self):
        assert self.mode == 'test'

        test_images = self.images
        test_labels = self.labels

        self.test_nums = test_images.shape[0]
        iter_num = self.test_nums

        iterator = iter([
            [
                test_images[i],
                test_labels[i]
            ] for i in range(iter_num)
        ])

        return iterator

    '''
    题目1:
        完成shuffle函数, 将 self.images, self.labels 两个numpy数组按照相同的顺序打乱
        其中 self.images 的维度为 [num_images, 784], self.labels 的维度为 [num_images]
    要求:
        只调用一次 np.random.shuffle(), 将两个数组按照相同的顺序打乱
    例子:
        无
    '''
    def shuffle(self):

        ### 请在下方编辑代码 ###
        index = np.arange(self.num_images)
        np.random.shuffle(index)
        self.images, self.labels = self.images[index], self.labels[index]
        ### 请在上方编辑代码 ###

    '''
    题目2:
        完成 normalize 函数, 对 self.images 数组中的每一张图片进行 normalization
        其中 self.images 的维度为 [num_images, 784]
    要求:
        normalization 后每张图片的数值范围都在 0 ~ 1 之间
    例子:
        无
    '''
    def normalize(self):
        ### 请在下方编辑代码 ###
        images_max = np.max(self.images, axis=-1, keepdims=True)
        images_min = np.min(self.images, axis=-1, keepdims=True)
        self.images = (self.images - images_min) / (images_max - images_min)
        ### 请在上方编辑代码 ###