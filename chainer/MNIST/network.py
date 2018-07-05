from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Model(Chain):
    def __init__(self, channel=1, c1=32, c2=64, f1=1024, n_out=10, filter_size1=3, filter_size2=3):
        super(Model, self).__init__(
            conv1=L.Convolution2D(channel,c1,filter_size1),
            conv2=L.Convolution2D(c1, c2, filter_size2),
            l1=L.Linear(None, f1),
            l2=L.Linear(None, n_out)
        )

    def __call__(self, x):
        x = x.reshape((len(x), 1, 28, 28))
        ##forward
        h=F.relu(self.conv1(x))
        h=F.max_pooling_2d(h,2)
        h=F.relu(self.conv2(h))
        h=F.max_pooling_2d(h,2)
        h=F.dropout(F.relu(self.l1(h)))
        y=self.l2(h)

        return y


