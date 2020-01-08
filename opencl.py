import numpy as np
import os
import math
import sys
from argparse import ArgumentParser
import pyopencl as cl

parser = ArgumentParser()
parser.add_argument("--mode", type=int, default=0, help="descrete:0, continuous:1")
parser.add_argument("--test", type=int, required=True, help="test number")
parser.add_argument("--train", type=int, required=True, help="train number")
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
    train_num = args.train
    test_num = args.test
    prior = calculate_prior(train_num, train_labels)

    n_local=1024
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg = cl.Program(ctx, """
    __kernel void discrete_likelihood(__global float* likelihood, __global float* images, __global float* labels, __global float* train_num)
    {
        int num = train_num[0];
        const int idx = get_global_id(0);
        if(idx>=num){ return; }
        for (int p=0; p<784; p++){
            int images_idx = idx*784 + p;
            int b = images[images_idx]/8;
            int lab = labels[idx]*784*32;
            int likelihood_idx = get_local_id(0)*10*784*32 + lab + p*32 + b;
            likelihood[likelihood_idx] += 1;
        }
        
    }

    __kernel void gaussian_likelihood_mean(__global float *mean, __global float* var, __global float *train_images, __global float *train_labels, __global float* train_num)
    {
        int num = train_num[0];
        const int idx = get_global_id(0);
        if(idx>=num){return;}
        for(int p=0; p<784; p++){
            int mean_idx = idx*10*784 + (int)train_labels[idx]*784 + p;
            mean[mean_idx] += train_images[idx*784+p];
        }
    }

    __kernel void gaussian_likelihood_var(__global float* mean, __global float* var, __global float* train_images, __global float* train_labels, __global float* train_num)
    {
        int num = train_num[0];
        const int idx = get_global_id(0);
        if(idx>=num){return;}

        for(int p=0; p<784; p++){
            int var_idx = idx*10*784 + (int)train_labels[idx]*784 + p;
            float m =  mean[ (int)train_labels[idx]*784+p ];
            float t = train_images[ idx*784+p ];
            var[var_idx] += (t-m)*(t-m);
        }
    }

    __kernel void discrete_predict(__global float *prob_record,__global float* error,__global float *prior,__global float* likelihood,__global float* test_images,__global float* test_labels, __global float* test_num)
    {
        int num = test_num[0];
        const int idx = get_global_id(0);
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

    __kernel void gaussian_predict(__global float *prob_record, __global float *error, __global float *prior, __global float *var, __global float *mean, __global float *test_images, __global float *test_labels, __global float* test_num)
    {
        int num = test_num[0];
        const int idx = get_global_id(0);
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
    """).build()
    
    # np.save("prior.npy", prior)
    # prior = np.load("prior.npy")

    test_images_np = test_images.astype('float32')
    test_labels_np = test_labels.astype('float32')
    train_images_np = train_images.astype('float32')
    train_labels_np = train_labels.astype('float32')
    train_num_np = np.zeros(1, dtype='float32')
    test_num_np = np.zeros(1, dtype='float32')
    train_num_np[0] = train_num
    test_num_np[0] = test_num
    train_num_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=train_num_np)
    test_num_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=test_num_np)


    # if args.mode==0: #DISCRETE
    bin_num = 32

    # ======================================
    # =========discrete_likelihood==========
    likelihood = np.zeros((n_local, class_num, pixels, bin_num), dtype='float32')

    likelihood_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=likelihood)
    train_images_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_images_np)
    train_labels_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_labels_np)
    
    N=train_num
    n_global = int( ( N + n_local - 1 ) / n_local )*n_local
    prg.discrete_likelihood(queue, (n_global,), (n_local,), likelihood_gpu, train_images_gpu, train_labels_gpu, train_num_gpu)
    
    cl.enqueue_copy(queue, likelihood, likelihood_gpu)
    likelihood_gpu.release()
    likelihood = likelihood.reshape(n_local, class_num, pixels, bin_num)
    likelihood = np.sum(likelihood, axis=0)
    # np.save("likelihood.npy", likelihood)
    # likelihood = np.load("likelihood.npy")
    # ======================================

    likelihood_div = np.sum(likelihood, axis=2)
    for c in range(class_num):
        for p in range(pixels):
            likelihood[c,p,:] = likelihood[c,p,:]/likelihood_div[c,p]
    likelihood = pseudo_count(likelihood.copy())

    # ===================================
    # =========DISCRETE_PREDICT==========
    likelihood_np = likelihood.astype('float32')
    error=np.zeros(test_num, dtype='float32')
    prob_record = np.zeros((test_num,10), dtype='float32')
    prior_np = prior.astype('float32')

    prob_record_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, prob_record.nbytes)
    error_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, error.nbytes)
    prior_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prior_np)
    likelihood_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=likelihood_np)
    test_images_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_images_np)
    test_labels_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_labels_np)

    N=test_num
    n_global = int( ( N + n_local - 1 ) / n_local )*n_local
    prg.discrete_predict(queue, (n_global,), (n_local,), prob_record_gpu, error_gpu, prior_gpu, likelihood_gpu, test_images_gpu, test_labels_gpu, test_num_gpu)
    
    cl.enqueue_copy(queue, prob_record, prob_record_gpu)
    cl.enqueue_copy(queue, error, error_gpu)
    prob_record_gpu.release()
    error_gpu.release()
    prior_gpu.release()
    likelihood_gpu.release()
    # test_images_gpu.release()
    # test_labels_gpu.release()
    # np.save("discrete_probability.npy", prob)
    # ===================================

    # show_result(prob_record, num=test_num)
    # show_discrete_imagination(likelihood)
    errorrate = np.sum(error)
    errorrate /= test_num
    print("\nError rate:", errorrate)
    
    # ===================================
    # ===================================
    # ===================================
    # else:
    # ===============gaussian likelihood===============
    mean = np.zeros((train_num*10*784), dtype='float32')
    var = np.zeros((train_num*10*784), dtype='float32')

    mean_gpu = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mean)
    var_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, var.nbytes)
    # train_images_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_images_np)
    # train_labels_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=train_labels_np)

    N=train_num
    n_global = int( ( N + n_local - 1 ) / n_local )*n_local
    prg.gaussian_likelihood_mean(queue, (n_global,), (n_local,), mean_gpu, var_gpu, train_images_gpu, train_labels_gpu, train_num_gpu)

    cl.enqueue_copy(queue, mean, mean_gpu)
    mean_gpu.release()
    mean = mean.reshape(train_num, 10, 784)
    mean = np.sum(mean, axis=0)
    for c in range(10):
        mean[c,:] = np.divide( mean[c,:], prior[c]*train_num)
    mean = mean.astype('float32')

    mean_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mean)

    prg.gaussian_likelihood_var(queue, (n_global,), (n_local,), mean_gpu, var_gpu, train_images_gpu, train_labels_gpu, train_num_gpu)
    cl.enqueue_copy(queue, var, var_gpu)
    mean_gpu.release()
    var_gpu.release()
    train_images_gpu.release()
    train_labels_gpu.release()
    var = var.reshape(train_num, 10, 784)
    var = np.sum(var, axis=0)
    for c in range(10):
        var[c,:] = np.divide( var[c,:], prior[c]*train_num )
    # np.save("mean.npy", mean)
    # np.save("var.npy", var)
    # ===================================

    # ==============gaussian predict=================
    var_np = var.astype('float32')
    error=np.zeros(test_num, dtype='float32')
    prob_record = np.zeros((test_num,10), dtype='float32')
    prior_np = prior.astype('float32')

    prob_record_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, prob_record.nbytes)
    error_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, error.nbytes)
    prior_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prior_np)
    var_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=var_np)
    mean_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mean)
    # test_images_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_images_np)
    # test_labels_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_labels_np)

    N=test_num
    n_global = int( ( N + n_local - 1 ) / n_local )*n_local
    prg.gaussian_predict(queue, (n_global,), (n_local,), prob_record_gpu, error_gpu, prior_gpu, var_gpu, mean_gpu, test_images_gpu, test_labels_gpu, test_num_gpu)
    
    cl.enqueue_copy(queue, prob_record, prob_record_gpu)
    cl.enqueue_copy(queue, error, error_gpu)
    prob_record_gpu.release()
    error_gpu.release()
    prior_gpu.release()
    var_gpu.release()
    mean_gpu.release()
    test_images_gpu.release()
    test_labels_gpu.release()
    # np.save("gaussian_probability.npy", prob)
    # ===================================

    # show_result(prob_record, num=test_num)
    # show_continuous_imagination(mean)
    errorrate = np.sum(error)
    errorrate /= test_num
    print("\nError rate:", errorrate)