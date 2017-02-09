'''
De Noising Auto Encoder
'''
from __future__ import print_function
import numpy as np
import numpy
import math
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy
import timeit
import inspect
import sys
import numpy
from theano.tensor.nnet import conv
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

import theano.tensor.nnet
import gc
gc.collect()

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

def drop(input, p=0.5):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied

    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.

    """
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask

class DropoutHiddenLayer(object):
    def __init__(self, is_train, rng, input=1, n_in=1, n_out=500, W=None, b=None,
                 activation=T.tanh, p=0.5):
        # type: (object, object, object, object, object, object, object, object, object) -> object
        """
        Hidden unit activation is given by: activation(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type is_train: theano.iscalar
        :param is_train: indicator pseudo-boolean (int) for switching between training and prediction

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type p: float or double
        :param p: probability of NOT dropping out a unit
        """
        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        output = activation(lin_output)

        # multiply output and drop -> in an approximation the scaling effects cancel out
        train_output = drop(output, p)

        # is_train is a pseudo boolean theano variable for switching between training and prediction
        self.output = T.switch(T.neq(is_train, 0), train_output, p * output)

        # parameters of the model
        self.params = [self.W, self.b]


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), poolonly=False):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        print('....image_shape....')
        print(image_shape)
        print('input shape....')
        print(filter_shape)
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode='half')
        # pool each feature map individually, using maxpooling
  

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

class PoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self,input,poolsize=(2, 2)):
        
     

        
        

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        return pooled_out


def train_nn(train_model,validate_model,test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs,
             verbose=True):
    """
    Wrapper function for training and test THEANO model

    :type train_model: Theano.function
    :param train_model:

    :type validate_model: Theano.function
    :param validate_model:

    :type test_model: Theano.function
    :param test_model:

    :type n_train_batches: int
    :param n_train_batches: number of training batches

    :type n_valid_batches: int
    :param n_valid_batches: number of validation batches

    :type n_test_batches: int
    :param n_test_batches: number of testing batches

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to

    """

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.85  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter % 100 == 0) and verbose:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)
           
            if (iter + 1) % validation_frequency == 0:
                
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch,
                       minibatch_index + 1,
                       n_train_batches,
                       this_validation_loss * 100.))
                if verbose:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           this_validation_loss * 100.))
               


                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_output = test_model(0)
                    f,axiss = plt.subplots(4, 2)
                    i = 0
                    #print (len(test_output))
                    while i < 8:
                        im=numpy.reshape(test_output[i],(3,32,32))
                        im = im.transpose(1,2,0)
                        plt.subplot(4,2,i+1)
                        plt.imshow(im)
                        i = i + 1
                    f.savefig('output/hw3a_{0}.png'.format(epoch, i))
                    plt.close(f)
                    # test it on the test set
                     
                   
                    
            
            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()

    # Retrieve the name of function who invokes train_nn() (caller's name)
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)

    # Print out summary
    print('Optimization complete.')
    print('Best validation error of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
import numpy
def floatX(X):
    return numpy.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(numpy.asarray(X, dtype=dtype), name=name)
def shared_zeros(shape, dtype=theano.config.floatX, name=None):
    return sharedX(numpy.zeros(shape), dtype=dtype, name=name)

def evaluate_lenet5(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, learning_rate=0.1,
                    n_epochs=128,

                    nkerns=[20, 50], batch_size=100):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)
    # train_set_x, train_set_y = datasets[0]
    # valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    
    
    x_reshaped = T.reshape(x, (batch_size, 3, 32, 32))
    layer0_input = T.reshape(x,(batch_size, 3, 32, 32))
    layer0_input = drop(layer0_input,0.7)
   
    #layer0_input = z.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    print(layer0_input.shape)

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(64, 3, 3, 3),
        poolsize=(1, 1)
    )

    print('layer 0 constructed....')
    print(layer0.output)
    

    
    print('layer 01 constructed....')
    # print(layer01.output)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        poolsize=(1, 1)
    )
    
 
    maxpool1 = theano.tensor.signal.pool.pool_2d(layer1.output, (2,2), ignore_border = True)
    
    layer3 = LeNetConvPoolLayer(
        rng,
        input=maxpool1,
        image_shape=(batch_size, 64, 16, 16),
        filter_shape=(128, 64, 3, 3),
        poolsize=(1, 1)

    )
    layer4 = LeNetConvPoolLayer(
        rng,
        input=layer3.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        poolsize=(1, 1)

    )
    
    maxpool2 = theano.tensor.signal.pool.pool_2d(layer4.output, (2,2), ignore_border=True)
    
    layer5 = LeNetConvPoolLayer(
        rng,
        input=maxpool2,
        image_shape=(batch_size, 128, 8, 8),
        filter_shape=(256, 128, 3, 3),
        poolsize=(1, 1)
    )
    #upsample1 = theano.tensor.extra_ops.repeat(layer5.output, 2,2)
    #upsample2 = theano.tensor.extra_ops.repeat(upsample1, 2,3)
    
    upsample1 = layer5.output.repeat(2,2)
    upsample2 = upsample1.repeat(2,3)
    
    layer6 = LeNetConvPoolLayer(
        rng,
        input=upsample2,
        image_shape=(batch_size, 256, 16, 16),
        filter_shape=(128, 256, 3, 3),
        poolsize=(1, 1)
    )

    layer7 = LeNetConvPoolLayer(
        rng,
        input=layer6.output,
        image_shape=(batch_size, 128, 16, 16),
        filter_shape=(128, 128, 3, 3),
        poolsize=(1, 1)
    )

    adder1 = layer7.output + layer3.output

    
    upsample3 = adder1.repeat(2, 2)
    print(upsample3.shape)
    upsample4 = upsample3.repeat(2, 3)
    print(upsample4.shape)
    layer8 = LeNetConvPoolLayer(
        rng,
        input=upsample4,
        image_shape=(batch_size, 128, 32, 32),
        filter_shape=(64, 128, 3, 3),
        poolsize=(1, 1)
    )
    
    layer9 = LeNetConvPoolLayer(
        rng,
        input=layer8.output,
        image_shape=(batch_size, 64, 32, 32),
        filter_shape=(64, 64, 3, 3),
        poolsize=(1, 1)

    )

    adder2 = layer9.output + layer1.output
    
    layer10 = LeNetConvPoolLayer(rng, input = adder2, image_shape = (batch_size, 64, 32, 32), filter_shape = (3, 64, 3,3), poolsize = (1,1))
    '''
    #theano.pp(layer10)
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    #layer3_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    
  

    # classify the values of the fully-connected sigmoidal layer
    

    # the cost we minimize during training is the NLL of the model
    
    #cost = numpy.mean(cost) 
    '''
    # create a function to compute the mistakes that are made by the model
    '''
    test_model = theano.function(
        #layer5.errors(y)
        [],cost,
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    

    validate_model = theano.function(
        #layer5.errors(y),
        [index],cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    '''
    '''
    # create a list of all model parameters to be fit by gradient descent
   

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    

    
   

    '''
    output =  layer10.output
    cost = T.mean((layer10.output - x_reshaped)**2)
    rho = 0.9
    lr = 0.001
    epsilon = 1e-8
    params = layer10.params + layer9.params + layer8.params + layer7.params + layer6.params + layer5.params + layer4.params + layer3.params + layer1.params + layer0.params
       
    grads = T.grad(cost, params)
    ac = [shared_zeros(p.get_value().shape) for p in params]
    updates = []

    for p, g, a in zip(params, grads, ac):
        new_a = rho * a + (1 - rho) * g ** 2 # update accumulator
        updates.append((a, new_a))

        new_p = p - lr * g / T.sqrt(new_a + epsilon)
        updates.append((p, new_p))
        
    validate_model = theano.function(
        #layer5.errors(y),
        [index],cost,
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )
    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
            #y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    test_model = theano.function(
        #layer5.errors(y),
        [index],output,
        givens={
            x: test_set_x[0+index: batch_size+index]
        }
    )
    train_nn(train_model,validate_model,test_model,
             n_train_batches, n_valid_batches, n_test_batches, n_epochs,
             verbose=True)
from hw3_utils import load_data

# downsample the training and validation dataset if needed, ds_rate should be larger than 1.
ds_rate = None
datasets = []
gc.collect()
datasets = load_data(ds_rate=ds_rate, theano_shared=True)
train_set_x, train_set_y = datasets[0]
print(train_set_x.get_value().shape)
import sys
#sys.exit()
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
evaluate_lenet5(train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y)
