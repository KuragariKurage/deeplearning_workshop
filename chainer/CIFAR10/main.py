import argparse
import chainer
from chainer import training
from chainer import iterators, optimizers
import chainer.links as L
from chainer.training import extensions
from network import Model
import numpy as np

def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        default=100,type=int)
    parser.add_argument("--epoch",
                        default=10, type=int)
    parser.add_argument("--learning_rate",
                        default=1e-08, type=float)
    parser.add_argument("--gpu_idx",
                        default=-1,type=int)
    return parser.parse_args()

def main():
    args=get_args()

    train_data=None
    train_label=[]
    for i in range(1,6):
        d=unpickle("cifar-10-batches-py/data_batch_%d" %i)
        if i==1:
            train_data=d['data']
        else:
            train_data=np.vstack((train_data,d['data']))
        train_label=train_label+d['labels']

    train_data=train_data.reshape((len(train_data),3,32,32))
    train_label=np.array(train_label)

    test_d=unpickle("cifar-10-batches-py/test_batch")
    test_data=test_d['data']
    test_data=test_data.reshape((len(test_data),3,32,32))
    test_label=np.array(test_d['labels'])

    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_data /= 255
    test_data /= 255
    train_label = train_label.astype(np.int32)
    test_label = test_label.astype(np.int32)

    train = chainer.datasets.tuple_dataset.TupleDataset(train_data,train_label)
    test = chainer.datasets.tuple_dataset.TupleDataset(test_data, test_label)

    #train, test = chainer.datasets.get_cifar10()
    minibatch_size = args.batch_size
    train_iter = iterators.SerialIterator(train, minibatch_size)
    test_iter = iterators.SerialIterator(test, minibatch_size, False, False)

    model = Model()

    if args.gpu_idx>=0:
        chainer.cuda.get_device(args.gpu_idx).use()
        model.to_gpu(args.gpu_idx)


    model=L.Classifier(model)

    optimizer=optimizers.Adam(eps=args.learning_rate)
    optimizer.setup(model)

    updater=training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu_idx)

    max_epoch=args.epoch

    trainer=training.Trainer(updater, (max_epoch, 'epoch'),out='cifar10_result')

    #save the log
    trainer.extend(extensions.LogReport())

    #save the model per 5 epoch
    trainer.extend(extensions.snapshot(), trigger=(5, 'epoch'))
    
    #validation
    trainer.extend(extensions.Evaluator(test_iter, model,device=args.gpu_idx))

    #print the log
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))	

    #save the glaph
    trainer.extend(extensions.dump_graph(root_name="main/loss", out_name="cg.dot"))	

    #save the loss glaph
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    #save the accuracy glaph
    trainer.extend(
        extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    #print the progress bar
    trainer.extend(extensions.ProgressBar())

    trainer.run()

if __name__ == '__main__':
    main()
