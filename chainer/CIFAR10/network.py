from chainer import Chain
import chainer.functions as F
import chainer.links as L

class Model(Chain):
    def __init__(self, channel=3, c1=32, c2=32,c3=64,c4=512, f1=1024, n_out=10, filter_size=3):
        super(Model, self).__init__(
            conv1=L.Convolution2D(channel,c1,filter_size),
            conv2=L.Convolution2D(None, c2, filter_size),
            conv3=L.Convolution2D(None, c3, filter_size),
            conv4=L.Convolution2D(None, c4, filter_size),
            l1=L.Linear(None, f1),
            l2=L.Linear(None, n_out)
        )

    def __call__(self, x):
        x = x.reshape((len(x), 3, 32, 32))
        ##forward
        h=F.relu(self.conv1(x))
        h=F.relu(self.conv2(h))
        h=F.max_pooling_2d(h,2)
        h = F.dropout(h)
        h=F.relu(self.conv3(h))
        h=F.relu(self.conv4(h))
        h=F.max_pooling_2d(h,2)
        h=F.dropout(F.relu(self.l1(h)))
        y=self.l2(h)
        return y


