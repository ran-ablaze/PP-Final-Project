import numpy as np
import os
# import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import math
import sys
from argparse import ArgumentParser

mod = SourceModule( """
    #include <math.h>
    #include <stdio.h>
    __global__ void discrete_likelihood(float* likelihood,  float* images, float* labels, float* train_num)
    {   

        int num = train_num[0];
        //printf("%d\\n", num);
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx>=num){ return; }
        for (int p=0; p<784; p++){
            int images_idx = idx*784 + p;
            int b = images[images_idx]/8;
            int lab = labels[idx]*784*32;
            int likelihood_idx = (threadIdx.x)*10*784*32 + lab + p*32 + b;
            likelihood[likelihood_idx] += 1;
        }
        
    }

    __global__ void gaussian_likelihood_mean(float *mean, float *train_images, float *train_labels, float* train_num)
    {
        int num = train_num[0];
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx>=num){return;}
        for(int p=0; p<784; p++){
            int mean_idx = idx*10*784 + (int)train_labels[idx]*784 + p;
            mean[mean_idx] += train_images[idx*784+p];
        }
    }

    __global__ void gaussian_likelihood_var(float* mean, float* var, float* train_images, float* train_labels, float* train_num)
    {
        int num = train_num[0];
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx>=num){return;}

        for(int p=0; p<784; p++){
            int var_idx = idx*10*784 + (int)train_labels[idx]*784 + p;
            float m =  mean[ (int)train_labels[idx]*784+p ];
            float t = train_images[ idx*784+p ];
            var[var_idx] += (t-m)*(t-m);
        }
    }

    __global__ void discrete_predict(float *prob_record, float* error, float *prior, float* likelihood, float* test_images, float* test_labels, float* test_num)
    {
        int num = test_num[0];
        //printf("%d\\n", num);
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx>=num){return;}
        float prob[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int c=0; c<10; c++){
            prob[c] += log(prior[c]);
            for (int p=0; p<784; p++){
                int test_images_idx = idx*784 + p;
                int likelihood_idx = c*(784*32) + p*32 + (int)(test_images[test_images_idx]/8);
                prob[c] += log( likelihood[likelihood_idx] );
                
            }
        }
        float sum_prob = 0.0;
        for(int i=0; i<10; i++){
            sum_prob += prob[i];
        }
        for(int i=0; i<10; i++){
            prob[i] /= sum_prob;
        }
        
        for(int i=0; i<10 ; i++){
            prob_record[idx*10+i]=prob[i];
        }
        
        int predict = 0;
        float min = 9999999;
        for(int i=0; i<10 ; i++){
            if(prob[i]<min)
            {
                min = prob[i];
                predict = i;
            }
        }
        if (predict != (int)test_labels[idx]) {error[idx] += 1;}
    }

    __global__ void gaussian_predict(float *prob_record, float *error, float *prior, float *var, float *mean, float *test_images, float *test_labels, float* test_num)
    {
        int num = test_num[0];
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx>=num){return;}
        float con_prob[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int c=0; c<10; c++){
            con_prob[c] += log(prior[c]);
            for (int p=0; p<784; p++){
                if(var[c*784 + p] == 0)
                    continue;
                con_prob[c] -= log(2.0*M_PI*var[c*784+p])/2.0;
                con_prob[c] -= pow((test_images[idx*784+p]-mean[c*784+p]),2)/(2.0*var[c*784+p]);
            }
        }
        float sum_prob = 0.0;
        for(int i=0; i<10; i++){
            sum_prob += con_prob[i];
        }
        for(int i=0; i<10; i++){
            con_prob[i] /= sum_prob;
        }
        for(int i=0; i<10 ; i++){
            prob_record[idx*10+i]=con_prob[i];
        }
        int predict = 0;
        float min = 9999999;
        for(int i=0; i<10 ; i++){
            if(con_prob[i]<min)
            {
                min = con_prob[i];
                predict = i;
            }
        }
        if (predict != (int)test_labels[idx]) {error[idx] += 1;}
    }
""" )


discrete_predict = mod.get_function("discrete_predict")
discrete_likelihood = mod.get_function("discrete_likelihood")
gaussian_predict = mod.get_function("gaussian_predict")
gaussian_likelihood_mean = mod.get_function("gaussian_likelihood_mean")
gaussian_likelihood_var = mod.get_function("gaussian_likelihood_var")
end = drv.Event()

parser = ArgumentParser()
parser.add_argument("--mode", type=int, default=0, help="descrete:0, continuous:1")
args = parser.parse_args()

def readmnist(mnist_dir, mode='training'):
    if mode == 'training':
        image_dir = os.path.join(mnist_dir, 'train-images-idx3-ubyte')
        label_dir = os.path.join(mnist_dir, 'train-labels-idx1-ubyte')
    elif mode == 'testing':
        image_dir = os.path.join(mnist_dir, 't10k-images-idx3-ubyte')
        label_dir = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte')

    with open(image_dir, 'rb') as fimage:
        #training: magic=2049, num=60000, row=28, col=28
        magic, num, row, col = np.fromfile(fimage, dtype=np.dtype('>i'), count=4)
        images = np.fromfile(fimage, dtype=np.dtype('>B'), count=-1)

    with open(label_dir, 'rb') as flabel: #read binary mode
        magic, num = np.fromfile(flabel, dtype=np.dtype('>i'), count=2)  #big endian, int
        labels = np.fromfile(flabel, dtype=np.dtype('>B'), count=-1) #unsigned byte, realall

    pixels = row*col
    images = images.reshape(num, pixels)

    return num, images, labels, pixels

def calculate_prior(train_num, train_labels):
    prior = np.zeros(class_num, dtype=float)
    for n in range(train_num):
        prior[train_labels[n]] += 1
    
    prior /= train_num  #is now a probability, sum to 1
    return prior

def pseudo_count(likelihood):
    for n in range(class_num):
        for p in range(pixels):
            for b in range(bin_num):
                if likelihood[n,p,b]==0:
                    likelihood[n,p,b] = 0.001
    return likelihood

def divide_marginal(prob):
    prob = np.divide(prob, sum(prob))
    return prob
'''
def show_discrete_imagination(likelihood):
    zero = np.sum(likelihood[:,:,0:16], axis=2)
    one = np.sum(likelihood[:,:,16:32], axis=2)
    
    im = np.zeros((class_num, pixels))
    im = (one >= zero)*1

    print("Imagination of numbers in Bayesian classifier:\n")
    for c in range(class_num):
        print("{}:".format(c))
        for row in range (28):
            for col in range(28):
                print(im[c][row*28+col], end=' ')
            print(" ")

def show_continuous_imagination(mean):
    im = np.zeros((class_num, pixels))
    im = (mean>=128)*1
    
    print("Imagination of numbers in Bayesian classifier:\n")
    for c in range(class_num):
        print("{}:".format(c))
        for row in range (28):
            for col in range(28):
                print(im[c][row*28+col], end=' ')
            print(" ")

def show_result(prob, num=10000):
    for n in range(num):
        print("Postirior (in log scale):")
        for i in range(class_num):
            print("{}: {}".format(i, prob[n][i]))
        print("Prediction: {}, Ans: {}\n".format(np.argmin(prob[n]), test_labels[n]))
        '''

if __name__=='__main__':
    mnist_dir = './data/'
    class_num = 10

    train_num, train_images, train_labels, pixels = readmnist(mnist_dir, 'training')
    test_num, test_images, test_labels, _ = readmnist(mnist_dir, 'testing')
    train_num = 60000 
    test_num = 10000 
    train_num_gpu = np.zeros(1, dtype='float32').ravel()
    test_num_gpu = np.zeros(1, dtype='float32').ravel()
    # print(train_num_gpu)
    train_num_gpu[0] = train_num
    test_num_gpu[0] = test_num

    train_images_gpu = train_images.ravel().astype('float32')
    train_labels_gpu = train_labels.ravel().astype('float32')
    test_images_gpu = test_images.ravel().astype('float32')
    test_labels_gpu = test_labels.ravel().astype('float32')

    prior = calculate_prior(train_num, train_labels)
    # np.save("prior.npy", prior)
    # prior = np.load("prior.npy")
    prior_gpu = prior.ravel().astype('float32')

    
    nThreads=1024
    bin_num = 32

    # if args.mode==0: #DISCRETE
        
    # ===================================
    # =========DISCRETE_PREDICT==========
    nBlocks = int( ( train_num + nThreads - 1 ) / nThreads )
    likelihood = np.zeros((nThreads, class_num, pixels, bin_num), dtype='float32').ravel()
    discrete_likelihood( drv.InOut(likelihood),  drv.In(train_images_gpu), drv.In(train_labels_gpu), drv.In(train_num_gpu),\
                        block=(nThreads,1,1), grid=(nBlocks,1) )
    likelihood = likelihood.reshape(nThreads, class_num, pixels, bin_num)
    likelihood = np.sum(likelihood, axis=0)
    # np.save("likelihood.npy", likelihood)
    # likelihood = np.load("likelihood.npy")
    # ===================================

    likelihood_div = np.sum(likelihood, axis=2)
    for c in range(class_num):
        for p in range(pixels):
            likelihood[c,p,:] = likelihood[c,p,:]/likelihood_div[c,p]
    likelihood = pseudo_count(likelihood.copy())

    # ===================================
    # =========DISCRETE_PREDICT==========
    likelihood_gpu = likelihood.ravel().astype('float32')
    nBlocks = int( ( test_num + nThreads - 1 ) / nThreads )
    error=np.zeros(test_num, dtype='float32')
    prob_record = np.zeros((test_num,10), dtype='float32')
    prob_record_gpu = prob_record.ravel()
    discrete_predict(drv.Out(prob_record_gpu), drv.Out(error),
                    drv.In(prior_gpu), drv.In(likelihood_gpu), drv.In(test_images_gpu), drv.In(test_labels_gpu), drv.In(test_num_gpu),
                    block=(nThreads,1,1), grid=(nBlocks,1) )
    # np.save("discrete_probability.npy", prob)
    # ===================================

    # show_result(prob, num=test_num)
    # show_discrete_imagination(likelihood)
    errorrate = np.sum(error)
    errorrate /= test_num
    print("\nError rate:", errorrate)

# else:
    # ===============gaussian likelihood===============
    nBlocks = int( ( train_num + nThreads - 1 ) / nThreads )
    mean_gpu = np.zeros((train_num*10*784), dtype='float32').ravel() #60000, 10, 784
    var_gpu = np.zeros((train_num*10*784), dtype='float32').ravel()
    gaussian_likelihood_mean(drv.InOut(mean_gpu), drv.In(train_images_gpu), drv.In(train_labels_gpu), drv.In(train_num_gpu),\
                            block=(nThreads,1,1), grid=(nBlocks,1) )
    mean = mean_gpu.reshape(train_num, 10, 784)
    mean = np.sum(mean, axis=0)
    for c in range(10):
        mean[c,:] = np.divide( mean[c,:], prior[c]*train_num)
    mean_gpu = mean.ravel().astype('float32')
    gaussian_likelihood_var(drv.In(mean_gpu), drv.Out(var_gpu), drv.In(train_images_gpu), drv.In(train_labels_gpu), drv.In(train_num_gpu),\
                            block=(nThreads,1,1), grid=(nBlocks,1) )
    var = var_gpu.reshape(train_num, 10, 784)
    var = np.sum(var, axis=0)
    for c in range(10):
        var[c,:] = np.divide( var[c,:], prior[c]*train_num )
    np.save('mean.npy', mean)
    np.save('var.npy', var)
    # ===================================

    # ==============gaussian predict=================
    var_gpu = var.ravel().astype('float32')
    mean_gpu = mean.ravel().astype('float32')
    nBlocks = int( ( test_num + nThreads - 1 ) / nThreads )
    error=np.zeros(test_num, dtype='float32')
    prob_record = np.zeros((test_num,10), dtype='float32')
    prob_record_gpu = prob_record.ravel()

    gaussian_predict(drv.Out(prob_record_gpu), drv.Out(error),
                    drv.In(prior_gpu), drv.In(var_gpu),drv.In(mean_gpu), drv.In(test_images_gpu), drv.In(test_labels_gpu), drv.In(test_num_gpu),
                    block=(nThreads,1,1), grid=(nBlocks,1) )
    # np.save("gaussian_probability.npy", prob)
    # ===================================

    # show_result(prob, num=test_num)
    # show_continuous_imagination(mean)
    errorrate = np.sum(error)
    errorrate /= test_num
    print("\nError rate:", errorrate)