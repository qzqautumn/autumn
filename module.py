'''
本任务输入输出说明:
    输入共5行:
        第1行: input_dim
        第2行: output_dim
        第3行: x
        第4行: lr
        第5行: grad
    输出共8行:
        input_dim 行: 初始化的 linear.w
        1 行: 初始化的 linear.b
        n 行 (n=3): y
        n 行 (n=3): dx
        input_dim 行: 更新后的 linear.w
        1 行: 更新后的 linear.b

注:
    本任务中, Linear 类的实现均需要考虑第一维为 batch 维度的情况
'''
import numpy as np

'''
题目1:
    补充 Linear
    Linear 具有一个初始化方法、两个属性和三个成员方法

    初始化方法:
        Linear 类的初始化方法 __init__() 需要在这里特别说明一下
        该初始化方法接受两个参数 input_dim 和 output_dim
        这两个参数决定了 self.w 和 self.b 的大小, 因此需要在 Linear 类被实例化时指定 input_dim 和 output_dim 具体是多少
        另外本任务中的可学习参数采用正态分布进行初始化, 对于mnist手写数据识别, 该方法完全适用

    属性: 
        self.w: 
            该属性是 linear 中的可学习参数, 维度为 [input_dim, output_dim]
        self.b:
            该属性是 linear 中的可学习参数, 维度为 [output_dim]

    成员方法:
        forward(): 
            该方法输入 x, 其为维度为 [batch_size, input_dim]
            该方法输出 y = x * w + b, y 的维度应当为 [batch_size, output_dim]
            x 和 y 将被保存在 self.x 和 self.y

        backward():
            该方法输入 grad, 其含义为 loss 对 y 的导数, 且 grad 和 y 具有相同的维度大小, 为 [batch_size, output_dim]
            在该方法中需要求出:
                loss 对 x 的导数 dx, 并将其作为返回值返回
                loss 对 w 的导数 dw, 并将其保存在 self.dw
                loss 对 b 的导数 dw, 并将其保存在 self.db

        updata():
            该方法输入 lr, 并需要使用随机梯度下降法更新 self.w 和 self.b

要求:
    根据上述说明将 Linear 类的三个成员方法补充完整, 使其能够处理一个 batch 的数据

例子:
    输入:
        input_dim: 4
        output_dim: 6
        x: [[1, 2 ,3 ,4 ], 
            [5, 6 ,7 ,8 ], 
            [9, 10,11,12]]
        lr: 0.001
        grad: [[0.321, 0.544, 0.678, 0.122, 0.572, 0.078],
               [0.41 , 0.54 , 0.68 , 0.72 , 0.22 , 0.84 ],
               [0.01 , 0.04 , 0.08 , 0.02 , 0.02 , 0.04 ]]

    输出:
        before update linear.w:
            [[-0.74625166  0.43412266 -0.74624712 -2.06427371  0.40335803 -0.06471956]
             [ 1.40454374 -1.55689742  0.22728222  0.31273043  0.26488845 -1.38061386]
             [-0.62488306  0.55796669 -2.80223066  0.13327384 -0.41106973  0.02707277]
             [-1.03668721 -0.25039667 -2.18639059  0.46541259  0.38649624  1.09460728]]

        before update linear.b:
            [0.05534438  0.06563665  1.18526187  0.69473411 -1.20362127 -0.84072899]
    
        y:
            [[-3.90321784e+00 -1.94172212e+00 -1.62586751e+01  1.51739313e+00  4.22893966e-02  7.92971174e-01]
             [-7.91633064e+00 -5.20254105e+00 -3.82890197e+01 -3.09403427e+00  2.61698132e+00 -5.01642276e-01]
             [-1.19294434e+01 -8.46335998e+00 -6.03193643e+01 -7.70546167e+00  5.19167325e+00 -1.79625573e+00]]

        linear.dw:
            [[0.82033333 1.20133333 1.59933333 1.30066667 0.61733333 1.546     ]
             [1.06733333 1.576      2.07866667 1.588      0.888      1.86533333]
             [1.31433333 1.95066667 2.558      1.87533333 1.15866667 2.18466667]
             [1.56133333 2.32533333 3.03733333 2.16266667 1.42933333 2.504     ]]

        linear.db:
            [0.247      0.37466667 0.47933333 0.28733333 0.27066667 0.31933333]

        dx:
            [[-0.53550833 -0.16001489 -2.01372678 -1.58812965]
             [-2.03088772 -0.98658404 -1.83215394 -0.7074052 ]
             [-0.08560448 -0.07372006 -0.21258162 -0.13447152]]

        after update linear.w:
            [[-0.74707199  0.43292133 -0.74784645 -2.06557437  0.4027407  -0.06626556]
             [ 1.4034764  -1.55847342  0.22520355  0.31114243  0.26400045 -1.38247919]
             [-0.6261974   0.55601602 -2.80478866  0.13139851 -0.4122284   0.0248881 ]
             [-1.03824854 -0.252722   -2.18942792  0.46324992  0.3850669   1.09210328]]
             
        after update linear.b:
            [0.05509738  0.06526198  1.18478254  0.69444678 -1.20389194 -0.84104833]
'''
class Linear():
    def __init__(self, input_dim, output_dim):

        self.w = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(output_dim)

        # self.w = np.random.randn(input_dim, output_dim)*np.sqrt(2/(output_dim+input_dim))
        # self.b = np.random.randn(output_dim)*np.sqrt(2/(output_dim+input_dim))

        # self.w = np.random.uniform(-np.sqrt(6/(output_dim+input_dim)),
        #                 np.sqrt(6/(output_dim+input_dim)),
        #                 size=(input_dim, output_dim))
        # self.b = np.zeros((1, output_dim))
        
    def forward(self, x):
        self.x = x
        self.batch_size = self.x.shape[0]

        ### 请在下方编辑代码 ###
        y = np.dot(x, self.w)
        y = y + self.b
        ### 请在上方编辑代码 ###

        self.y = y
        return y
    
    def backward(self, grad):

        ### 请在下方编辑代码 ###
        dw = np.dot(self.x.T, grad)
        db = np.sum(grad, axis=0)
        dx = np.dot(grad, self.w.T)
        ### 请在上方编辑代码 ###

        self.dw = dw
        self.db = db
        return dx

    def update(self, lr):

        ### 请在下方编辑代码 ###
        self.w = self.w - lr * self.dw / self.batch_size
        self.b = self.b - lr * self.db / self.batch_size
        ### 请在上方编辑代码 ###