'''
本任务输出说明:
    输入共2行:
        第1行: x
        第2行: grad
    输出共6行
        第1行: sigmoid.forward(x)
        第2行: sigmoid.backward(grad)
        第3行: tanh.forward(x)
        第4行: tanh.backward(grad)
        第5行: softmax.forward(x)
        第6行: softmax.backward(grad)
'''
import numpy as np

'''
题目1:
    Sigmoid 类具有两个方法, 分别为负责前向推导的 forward() 和负责反向传播的 backward()
    
    forward() 方法的具体实现已经给出:
        forward() 的方法的输入为 x, 输出为 y=sigmoid(x)
        输入 x 和输出 y 将被 self.x 和 self.y 两个属性记录, 
        需要时 backward() 方法会使用 self.x 或 self.y 来计算梯度
    
    backward() 方法的输入为 grad, 其含义为 loss 对 y 的导数, 
    根据链式法则, 只要求出 y 对 x 的导数, 就可以得到 loss 对 x 的导数

    注: x 和 grad 为维度相同的二维向量, 维度均为 [1, dim]
要求:
    根据链式法则及 Sigmoid 的导数公式,
    补充 backward() 方法, 计算 loss 对输入 x 的导数 dx
例子:
    x = [[1 2 3]]
    y = sigmoid.forward(x) = [[0.73105858 0.88079708 0.95257413]]
    grad = [[0.1 0.2 0.3]]
    dx = sigmoid.backward(grad) = [[0.01966119 0.02099872 0.013553  ]]
'''
class Sigmoid():

    def forward(self, x):
        self.x = x
        y = 1.0 / (1.0 + np.exp(-x))
        self.y = y
        return self.y

    def backward(self, grad):

        ### 请在下方编辑代码 ###
        dx = np.multiply(grad, np.multiply(self.y, 1-self.y))
        ### 请在上方编辑代码 ###

        return dx

'''
题目2:
    Tanh 类具有两个方法, 分别为负责前向推导的 forward() 和负责反向传播的 backward()
    
    forward() 的方法的输入为 x, 输出为 y=tanh(x)
    输入 x 和输出 y 用 self.x 和 self.y 两个属性记录
    需要时 backward() 方法会使用 self.x 或 self.y 来计算梯度

    backward() 方法的具体实现已经给出:
        backward() 方法的输入为 grad, 其含义为 loss 对 y 的导数, 
        根据链式法则, 只要求出 y 对 x 的导数, 就可以得到 loss 对 x 的导数

    注: x 和 grad 为维度相同的二维向量, 维度均为 [1, dim]
要求:
    根据 tanh 的公式补充 forward() 函数, 
    并将输入 x 和输出 y 用 self.x 和 self.y 两个属性记录
例子:
    x = [[1 2 3]]
    y = tanh.forward(x) = [[0.76159416 0.96402758 0.99505475]]
    grad = [[0.1 0.2 0.3]]
    dx = tanh.backward(grad) = [[0.04199743 0.01413016 0.00295981]]
'''
class Tanh():
    def forward(self, x):
        self.x = x
        ### 请在下方编辑代码 ###
        y = 2.0 / (1.0 + np.exp(-2*self.x)) - 1.0
        ### 请在上方编辑代码 ###
        self.y = y
        return y

    def backward(self, grad):

        dx = np.multiply(grad, 1 - np.multiply(self.y, self.y))
        
        return dx

'''
题目3:
    Softmax 类具有两个方法, 分别为负责前向推导的 forward() 和负责反向传播的 backward()

    forward() 的方法的输入为 x, 输出为 y=softmax(x)
    输入 x 和输出 y 用 self.x 和 self.y 两个属性记录
    需要时 backward() 方法会使用 self.x 或 self.y 来计算梯度

    backward() 方法的输入为 grad, 其含义为 loss 对 y 的导数, 
    根据链式法则, 只要求出 y 对 x 的导数, 就可以得到 loss 对 x 的导数

    注: x 和 grad 为维度相同的二维向量, 维度均为 [1, dim]
要求:
    根据 softmax 及其导数公式补充 forward() 与 backward() 方法
例子:
    # x = [[1 2 3]]
    # y = softmax.forward(x) = [[0.09003057 0.24472847 0.66524096]]
    # grad = [[0.1 0.2 0.3]]
    # dx = softmax.backward(grad) = [[ 0.03083358 -0.00900439 -0.1172952 ]]
'''
class Softmax():

    def forward(self, x):
        self.x = x
        ### 请在下方编辑代码 ###
        exp_x = np.exp(x)
        y = exp_x / exp_x.sum(axis=-1, keepdims=True)
        ### 请在上方编辑代码 ###
        self.y = y
        return y

    def backward(self, grad):
        ### 请在下方编辑代码 ###
        B, dim = self.x.shape
        dx = np.zeros((B, dim))
        for b in range(B):
            dx_curr = np.diag(self.y[b]) - np.outer(self.y[b], self.y[b])
            dx[b] = np.dot(grad[b], dx_curr)
        ### 请在上方编辑代码 ###
        return dx