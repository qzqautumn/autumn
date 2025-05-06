import numpy as np
from dataset import DataGenerator
from model import Classifier

'''
题目1:
    在字典 'hp' 中补充创建模型所需要的超参数
'''
### 请在下方编辑代码 ###
hp = {
    "lr": 1e-1,
    "batch_size": 16,
    "valid_ratio": 0.1,

    "epochs": 40,

    "input_dim": 784,
    "hidden_dim_1": 64,
    "hidden_dim_2": 16,
    "output_dim": 10,
}
### 请在上方编辑代码 ###

### 创建训练数据生成器
train_data_generator = DataGenerator(
        mode="train", 
        batch_size=hp["batch_size"], 
        vaild_ratio=hp["valid_ratio"]
    )
train_data_generator.load_data()

### 由于系统限制, 这里我们只使用 mnist 训练集中的前 10000 组数据
train_data_generator.images = train_data_generator.images[:10000]
train_data_generator.labels = train_data_generator.labels[:10000]
train_data_generator.num_images = 10000

train_data_generator.normalize()

### 创建测试数据生成器
test_data_generator = DataGenerator(mode="test", batch_size=1)
test_data_generator.load_data()

### 由于系统限制, 这里我们只使用 mnist 测试集中的前 1000 组数据
test_data_generator.images = test_data_generator.images[:1000]
test_data_generator.labels = test_data_generator.labels[:1000]
test_data_generator.num_images = 1000

test_data_generator.normalize()

### 创建分类器模型
classifier = Classifier(hp)

### 开始训练与验证, 每个 epoch 验证一次
for epoch in range(hp["epochs"]):

    '''
    题目2:
        补充下方代码, 完成训练过程
    '''
    train_data_generator.shuffle()
    train_iter = train_data_generator.getTrainIterator()
    for images, labels in train_iter:
        ### 请在下方编辑代码 ###
        loss, pred = classifier.forward(images, labels)
        classifier.backward()
        classifier.update()
        ### 请在上方编辑代码 ###
    print("[epoch {}] [training ...]".format(epoch))

### 开始测试
'''
题目4:
    补充下方代码, 完成一次测试过程, 并计算模型在测试集上对手写数字识别的准确率
'''
### 请在下方编辑代码 ###
test_acc = 0
test_iter = test_data_generator.getTestIterator()
for images, labels in test_iter:
    loss, pred = classifier.forward(images.reshape(1, -1), labels)
    pred = pred.argmax(axis=1)
    if pred == labels:
        test_acc += 1
test_acc = test_acc / test_data_generator.test_nums
### 请在上方编辑代码 ###
# print(test_acc)
print("[test] acc: {}".format(test_acc))