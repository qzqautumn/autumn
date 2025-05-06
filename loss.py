'''
本任务输入输出说明:
    输入共2行:
        第1行: logits
        第2行: labels
    输出共 3 * logits.shape[0] 行:
        logits.shape[0] 行: one_hot_labels
        logits.shape[0] 行: p
        logits.shape[0] 行: d_logits

注:
    本任务中, softmax 激活函数和 CrossEntropyLoss 的实现均需要考虑第一维为 batch 维度的情况
'''
import numpy as np

'''
题目1:
    请根据 softmax 公式补充 Softmax 类的 forward() 方法

要求:
    forward() 方法的输入为 x, 输出为 y

    输入 x 是一个二维矩阵, 其维度为 [batch_size, num_classes]
    其含义为 x 中有 batch_size 个样本, 每个样本含有 num_classes 个元素

    要求对每一个样本都实现 softmax 激活函数的计算

例子:
    x = [[-1.7537488 , -0.32080537, -0.73467751, -0.77535831],
         [ 0.19776362, -2.23367828,  1.09301109,  1.03164135]]

    y = softmax.forward(x) = [[0.09414592, 0.39456717, 0.26084263, 0.25044428],
                              [0.17128918, 0.01505803, 0.41930589, 0.3943469 ]]

    y.sum(axis=1) = [1., 1.]
'''
class Softmax():

    def forward(self, x):

        ### 请在下方编辑代码 ###
        exp_x = np.exp(x)
        y = exp_x / exp_x.sum(axis=1, keepdims=True)
        ### 请在上方编辑代码 ###

        return y


'''
题目2:
    补充 CrossEntropyLoss 类
    CrossEntropyLoss 具有一个属性和三个方法

    属性: 
        self.softmax: 
            该属性是上面的 Softmax 类实例化后的一个对象, 用于在计算交叉熵之前应用 Softmax 激活函数
    方法:
        one_hot(): 
            该方法输入 labels 和 num_classes
            labels 是标签, 维度为 [batch_size]
            num_classes 是分类总数
            该方法用于对 labels 进行 one-hot 编码, 返回 one_hot_labels, 其维度为 [batch_size, num_classes]

        forward(): 
            该方法输入 logits 和 labels
            logits 是前级网络的输出, 也是交叉熵损失函数的输入, 维度为 [batch_size, num_classes]
            labels 是标签, 维度为 [batch_size]
            该方法用于计算交叉熵损失:
                该方法首先调用 one_hot() 方法对 labels 进行 one-hot 编码, 得到 one_hot_labels
                再调用类属性 self.softmax 的 forward() 方法对 logits 进行激活, 得到 p
                one_hot_labels 和 p 将由 self.one_hot_labels 和 self.p 记录, 如有需要将在 backward() 中被使用
                最后根据交叉熵损失函数公式计算交叉熵, 输出 loss
        backward():
            该方法用于计算 loss 对 logits 的导数

要求:
    根据上述说明将 CrossEntropyLoss 类的三个方法补充完整, 使其能够对一个 batch 的数据和标签计算损失并梯度回传

例子:
    输入:
        logits = [[ 0.74302869,  0.37987365,  0.33879474, -0.57732144],
                  [-0.63449653, -1.85659425,  0.62052232,  0.17054752]]

        labels = [0, 3]

    输出:
        one_hot_labels = [[1., 0., 0., 0.],
                          [0., 0., 0., 1.]]

        p = softmax.forward(x) = [[0.38022693, 0.26443969, 0.25379689, 0.10153648],
                                  [0.14205936, 0.04185232, 0.49833066, 0.31775766]]

        loss = 1.056726631064392

        d_logits = [[-0.61977307,  0.26443969,  0.25379689,  0.10153648],
                    [ 0.14205936,  0.04185232,  0.49833066, -0.68224234]]
'''
class CrossEntropyLoss():

    def __init__(self):

        self.softmax = Softmax()

    def one_hot(self, labels, num_classes):

        ### 请在下方编辑代码 ###
        one_hot_labels = np.eye(num_classes)[labels]
        ### 请在上方编辑代码 ###

        return one_hot_labels

    def forward(self, logits, labels):

        self.logits = logits
        batch_size, num_classes = logits.shape

        one_hot_labels = self.one_hot(labels, num_classes)
        p = self.softmax.forward(self.logits)

        self.batch_size = batch_size
        self.labels = one_hot_labels
        self.p = p

        ### 请在下方编辑代码 ###
        loss = np.sum(-one_hot_labels * np.log(p)) / self.batch_size
        ### 请在上方编辑代码 ###
        
        return loss, p

    def backward(self):

        ### 请在下方编辑代码 ###
        d_logits = (self.p - self.labels)
        ### 请在上方编辑代码 ###

        return d_logits