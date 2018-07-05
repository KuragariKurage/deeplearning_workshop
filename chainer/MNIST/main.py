import argparse
import chainer
from chainer import training
from chainer import iterators, optimizers
import chainer.links as L
from chainer.training import extensions
from network import Model

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
    train, test = chainer.datasets.get_mnist()

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

    trainer=training.Trainer(updater, (max_epoch, 'epoch'),out='mnist_result')

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
