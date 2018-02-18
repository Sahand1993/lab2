
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter('ignore')
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

n = int(2*np.pi//0.1)

def gen_datasets():
    n = 2*np.pi//0.1

    x_train = np.linspace(0, 2*np.pi-0.1, n)
    x_test = np.linspace(0.05, 2*np.pi+0.05-0.1, n)

    ysin_train = np.sin(2*x_train)
    ysin_test = np.sin(2*x_test)

    ysquare_train = signal.square(2*x_train)
    ysquare_test = signal.square(2*x_test)

    return {"x_train":x_train, "ysin_train":ysin_train, "ysquare_train":ysquare_train, "x_test":x_test, "ysin_test":ysin_test, "ysquare_test":ysquare_test}

def add_noise(points):
    var = 0.1 
    noise = np.random.normal(0, np.sqrt(var), points.shape)
    points = points+noise
    return points

class RBF(object):
    def __init__(self, no_of_nodes, means, stds, weights, eta = 0.001):

        if no_of_nodes != len(means) or no_of_nodes != len(stds):
            raise ValueError("Number of nodes, number of means and number of standard deviations must be equal")

        self.means = means
        self.stds = stds
        self.no_of_nodes = no_of_nodes
        self.weights = weights
        self.eta = eta
    
    def comp_transfers(self, pattern):        
        return np.exp((-(pattern-self.means)*(pattern-self.means))/(2*self.stds*self.stds))

    def output(self, pattern):
        transfers = self.comp_transfers(pattern)
        return np.dot(transfers, self.weights)

    def abs_res_error(self, patterns, targets):
        """Average absolute value of error for the whole dataset."""
        output = np.vectorize(self.output)
        outputs = output(patterns)

        errors = np.abs(outputs-targets)
        avg_error = np.sum(errors)/errors.shape[0]

        return avg_error
    
    def abs_res_error_thresh_square(self, patterns, targets):
        output = np.vectorize(self.output)
        outputs = output(patterns)
        for i in range(outputs.shape[0]):
            outputs[i] = 1 if outputs[i] > 0 else -1
        errors = np.abs(outputs-targets)
        
    def outputs(self, patterns):
        output = np.vectorize(self.output)
        outputs = output(patterns)
        return outputs

    def gated_outputs(self, patterns):
        outputs = self.outputs(patterns)
        gate = np.vectorize(lambda x: 1 if x > 0 else -1)
        gated_outputs = gate(outputs)
        return gated_outputs

    def batch_train(self, patterns, targets):
        """trains the network with the least squares approximation for all patterns and targets"""
        fi = self.comp_transfers(patterns[0])
        for index, pattern in enumerate(patterns):
            if index == 0:
                continue
            fi = np.vstack((fi, self.comp_transfers(pattern)))

        a = np.dot(fi.T, fi)
        b = np.dot(fi.T, targets)
        self.weights = np.linalg.lstsq(a,b, rcond=None)[0]

    def instant_error(self, pattern, target):
        """Returns the instantaneous error for an input value"""
        instant_error = 0.5*(self.output(pattern)-target)**2
        return instant_error
    
    def delta_weights_online(self, pattern, target):
        instant_error = self.instant_error(pattern, target)
        transfers = self.comp_transfers(pattern)
        delta_w = self.eta*instant_error*transfers
        return delta_w
    
    def update_weights_online(self, pattern, target):
        delta_w = self.delta_weights_online(pattern, target)
        self.weights -= delta_w
        
def means4():
    means = np.array([
        np.pi/4,
        3*np.pi/4,
        5*(np.pi/4),
        7*(np.pi/4),
        ])
    return means

def get_means(nodesperextr):
    d = np.pi/(2*(nodesperextr+1))
    means = []
    for i in range(4):
        base = i*np.pi/2
        for j in range(nodesperextr):
            means.append(base+d*(j+1))

    return np.array(means)


def get_stds(noofnodes, std):
    stds = np.ones(noofnodes)
    stds = stds*std
    return stds

def print_gated_outputs(rbf, patterns, targets):
    plt.plot(patterns, targets)
    outputs = rbf.outputs(patterns)
    gated_outputs = rbf.gated_outputs(patterns)
    if targets.shape != outputs.shape:
        raise Exception("Mismatch")
    print("targets, gated outputs,  outputs")
    for i in range(targets.shape[0]):
        print("{0:4}   {1:4},             {2:4}".format(targets[i], gated_outputs[i], outputs[i]))
        if targets[i] != gated_outputs[i]:
            plt.plot(patterns[i], targets[i], "yx")

def print_outputs(rbf, patterns, targets):
    plt.plot(patterns, targets, "b.", label="targets")
    outputs = rbf.outputs(patterns)
    plt.plot(patterns, outputs, "r.", label="outputs")
    plt.legend()
    if targets.shape != outputs.shape:
        raise Exception("mismatch")
    print("Targets,   Outputs")
    for i in range(targets.shape[0]):
        print("{0:4} {1:4}".format(targets[i], outputs[i]))


def ls_errors_square(std, n):
    """Prints the absolute residual errors from 1 up to upto nodes per pi/2 for least squares batch training."""
    datasets = gen_datasets()
    patterns_train = datasets["x_train"]
    targets_train = datasets["ysquare_train"]
    patterns_test = datasets["x_test"]
    targets_test = datasets["ysquare_test"]
    outputs_test = []
    errors = []
    for i in range(n):
    
    
        means = get_means(i+1)
        nodes = means.shape[0]
        stds = get_stds(nodes, std)
        weights = np.ones(nodes)
        rbf = RBF(nodes, means, stds, weights)

    
        rbf.batch_train(patterns_train, targets_train)
        if i==0:
            outputs_test.append(rbf.outputs(patterns_test))
            errors.append(rbf.abs_res_error(patterns_test, targets_test))
            continue

        outputs_test.append(rbf.outputs(patterns_test))
        errors.append(rbf.abs_res_error(patterns_test, targets_test))

    i = 1 
    for output in outputs_test:
        fig = plt.figure()
        fig.suptitle("{0} nodes".format(i*4))
        means = get_means(i) 
        fig.text(0,0, s="error: {0}".format(errors[i-1]))
        plt.plot(means, np.zeros(means.shape), "go")
        plt.plot(patterns_test, output, "r.", label="outputs")
        plt.plot(patterns_test, targets_test, "b.", label="targets")
        plt.legend()
        i+=1

    plt.show()
    
def pt31():
    """Calculating absolute residual errors for 4 to 4*n RBFs with least squares batch training. No gating."""
    n = 10
    ls_errors_square(0.1, n)

def pt32(n):
    """least squares batch training with noise for 4 to n*4 nodes with n nodes per pi/2, none at the borders."""

    datasets = gen_datasets()


    datasets["ysin_train"] = add_noise(datasets["ysin_train"])
    datasets["ysin_test"] = add_noise(datasets["ysin_test"])

    patterns_train = datasets["x_train"]
    targets_train = datasets["ysin_train"]
    patterns_test = datasets["x_test"]
    targets_test = datasets["ysin_test"]

    outputs_test = []
    errors = []

    for i in range(n):

        means = get_means(i+1)
        nodes = means.shape[0]

        std = 0.5
        stds = get_stds(nodes, std)

        weights = np.ones(nodes)

        rbf = RBF(nodes, means, stds, weights)

        rbf.batch_train(patterns_train, targets_train)

        if i==0:
            outputs_test.append(rbf.outputs(patterns_test))
            errors.append(rbf.abs_res_error(patterns_test, targets_test))
            continue

        outputs_test.append(rbf.outputs(patterns_test))
        errors.append(rbf.abs_res_error(patterns_test, targets_test))

    i = 1
    for output in outputs_test:
        fig = plt.figure()
        fig.suptitle("{0} nodes".format(i*4))
        means = get_means(i)
        fig.text(0,0, s="error: {0}".format(errors[i-1]))
        plt.plot(means, np.zeros(means.shape), "go")
        plt.plot(patterns_test, output, "r.", label="outputs")
        plt.plot(patterns_test, targets_test, "b.", label="targets")
        plt.legend()
        i+=1
    plt.show()


# <h2>3.1 Batch mode training using least squares</h2>

# Let's first use our RBF ANN implementation to approximate a square sine wave.

# In[2]:




# In green we have the placement of the nodes on the x axis. Blue is the datapoints for the square sine wave. Red is our ANN output for each point. We can see that a higher number of nodes gives us better results. But even with 32 nodes we cannot get an average residual error lower than 0.24. Let's see what we get for an even higher number of nodes.

# In[3]:


def create_RBF_network(n, std):
    """Creates an RBF network with n*4 nodes."""
    means = get_means(n)
    nodes = means.shape[0]
    stds = get_stds(nodes, std)
    weights = np.ones(nodes)
    rbf = RBF(nodes, means, stds, weights)
    return rbf



# <p>We see an even lower error and better fit to the data.</p>

# In[4]:



# But, doubling the amount of nodes from there does not result in a considerable error reduction.
# 
# <h4>Trying sin(2x)</h4>
# <p>Let's try the same thing for sin(2x) now. We generate a dataset with based on the sin(2x) function with the test set x-values 0.05  (half the step size) higher that the training set x-values.</p>

# In[5]:


def sin32(n, std):
    datasets = gen_datasets()

    patterns_train = datasets["x_train"]
    targets_train = datasets["ysin_train"]
    patterns_test = datasets["x_test"]
    targets_test = datasets["ysin_test"]
    
    outputs_test = []
    errors = []
    for i in range(n):

        print("{0} NODES, std={1}".format((i+1)*4, std))

        means = get_means(i+1)
        nodes = means.shape[0]
        stds = get_stds(nodes, std)
        weights = np.ones(nodes)
        rbf = RBF(nodes, means, stds, weights)


        rbf.batch_train(patterns_train, targets_train)
        errors.append(rbf.abs_res_error(patterns_test, targets_test))
        if i==0:
            outputs_test.append(rbf.outputs(patterns_test))
            print("ERROR ({0} nodes): {1}\n".format(nodes, str(rbf.abs_res_error(patterns_test, targets_test))))
            continue

        outputs_test.append(rbf.outputs(patterns_test))
        print("ERROR ({0} nodes): {1}\n".format(nodes, str(rbf.abs_res_error(patterns_test, targets_test))))

    i = 1
    for output in outputs_test:
        fig = plt.figure()
        fig.suptitle("{0} nodes".format(i*4))
        fig.text(0,0,s="error: {0}".format(errors[i-1]))
        means = get_means(i)
        plt.plot(means, np.zeros(means.shape), "go")
        plt.plot(patterns_test, output, "r.", label="outputs")
        plt.plot(patterns_test, targets_test, "b.", label="targets")
        plt.legend()
        i+=1

    plt.show()



# <p>Clearly, the sine function is far more amenable to RBF ANN approximation. The error drops to virtually zero at 16 nodes with our distribution.</p>
# 
# <h2>Introducing noise to sin(2x)</h2>
# 
# Let's add some noise to the sine function and see what happens.

# In[6]:



# <p>Here one can notice the overfitting that takes places with the addition of nodes. At 32 nodes, our network output looks more noisy than our output with 4 nodes. This means poorer generalization, and unsurprisingly, the error is higher for 32 nodes than for 4 nodes. In fact, 4 nodes generated the lowest error of all networks.</p>
# 
# <h3>Trying sequential learning</h3>

# In[ ]:


def online_rbf(n, std, max_epochs):
    """
    Train RBF ANNs with 4 to n*4 nodes for 1 to 
    max_epochs epochs with sequential learning and
    print the absolute residual errors.
    """
    datasets = gen_datasets()
    patterns_train = datasets["x_train"]
    targets_train = datasets["ysin_train"]
    patterns_test = datasets["x_test"]
    targets_test = datasets["ysin_test"]
    dataset_train = np.vstack((patterns_train, targets_train))
    
    for i in range(n):
        #use i*4 number of nodes
        
        for j in range(max_epochs):
            #train j number of epochs
            rbf = create_RBF_network(i+1, std)
            #train one epoch
            
            #shuffle weights
            patterns_train = np.copy(patterns_train)
            targets_train = np.copy(targets_train)
            dataset_train = np.vstack((patterns_train, targets_train))
            #shuffle dataset_train and then separate into shuffled_patterns_train and shuffled_targets_train)
            np.random.shuffle(dataset_train.T)
            shuffled_patterns_train = dataset_train[0,:]
            shuffled_targets_train = dataset_train[1,:]
            #plt.figure()
            #plt.plot(shuffled_patterns_train, shuffled_targets_train, "r.")
            #for each datapoint
            for pattern, target in zip(shuffled_patterns_train, shuffled_targets_train):
                rbf.update_weights_online(pattern, target)
                error = rbf.abs_res_error(patterns_test, targets_test)

                plt.plot(j+1, error, "r.", label="error")

            
            
            if (j+1)%100 == 0:
                print("{0} error after {1} epochs.".format(error, j+1))





online_rbf(1, 1, 300)
plt.legend()
plt.show()

