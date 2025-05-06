from activation import Sigmoid, Tanh
from loss import CrossEntropyLoss
from module import Linear

class Classifier():
    def __init__(self, hp):

        self.hp = hp
        
        '''
        题目1:
            利用字典 'hp' 中的超参数创建模型中的可学习部分
        '''
        ### 请在下方编辑代码 ###
        self.layer_1 = Linear(self.hp["input_dim"], self.hp["hidden_dim_1"])
        self.layer_2 = Linear(self.hp["hidden_dim_1"], self.hp["hidden_dim_2"])
        self.layer_3 = Linear(self.hp["hidden_dim_2"], self.hp["output_dim"])
        ### 请在上方编辑代码 ###

        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        self.cross_entropy_loss = CrossEntropyLoss()

    '''
    题目2:
        将模型的前向推导过程补充完整
        forward() 方法将输出 计算所得 loss 和 模型的输出
    '''
    def forward(self, x, y):

        ### 请在下方编辑代码 ###
        h_1 = self.layer_1.forward(x)
        a_1 = self.sigmoid.forward(h_1)

        h_2 = self.layer_2.forward(a_1)
        a_2 = self.tanh.forward(h_2)

        h_3 = self.layer_3.forward(a_2)
        loss, p = self.cross_entropy_loss.forward(h_3, y)
        ### 请在上方编辑代码 ###

        return loss, p

    '''
    题目3:
        将模型的反向传播过程补充完整
        该方法无返回值
    '''
    def backward(self,):
        ### 请在下方编辑代码 ###
        d_h_3 = self.cross_entropy_loss.backward()
        d_a_2 = self.layer_3.backward(d_h_3)

        d_h_2 = self.tanh.backward(d_a_2)
        d_a_1 = self.layer_2.backward(d_h_2)

        d_h_1 = self.sigmoid.backward(d_a_1)
        d_x = self.layer_1.backward(d_h_1)
        ### 请在上方编辑代码 ###

    '''
    题目4:
        将模型的参数更新过程补充完整
        该方法无返回值
    '''
    def update(self):
        lr = self.hp["lr"]
        ### 请在下方编辑代码 ###
        self.layer_1.update(lr)
        self.layer_2.update(lr)
        self.layer_3.update(lr)
        ### 请在上方编辑代码 ###