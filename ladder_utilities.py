
# coding: utf-8

# # Collection of functions used in fc ladder 
# 
# Architecture and formulas for the fc ladder layers as described in https://arxiv.org/abs/1511.06430.
# 
# ### Policy: only edit functions in current *.ipynb, than save as *.py from which we import the functions

# In[1]:

'''
# used docker command on cluster (possilbly data in Oliver’s data directory):

nvidia-docker run -it -p 8710:8888 -p 8711:6006 -v /cluster/home/sick/:/notebooks/local -v /cluster/data/dueo/:/data -it oduerr/tf_docker:tf1_gpu_py3
'''

'''
# if working on laptop on local docker:
docker run -p 4242:8888 -v ~/dl_cas/:/notebooks -p 6006:6006 -it oduerr/tf_docker:tf1_py3
'''


# ## Imports

# In[2]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import numpy as np
import time
import tensorflow as tf
import pandas as pd
tf.set_random_seed(1)
np.random.seed(1)
import sys
tf.__version__, sys.version_info


# ### convertToOneHot

# In[3]:

##############################################################
# function that coverts a list of integer to an array of corresponding 1-hot encoding rows with length num_class
# function call: convertToOneHot(vector, num_classes=None)
# input: vector (list of integer), num_class (number of class labels)
# output: a numpy array of corresponding 1-hot encoded classes, each row has length num_classes
# Taken from http://stackoverflow.com/questions/29831489/numpy-1-hot-array
def convertToOneHot(vector, num_classes=None):
    # zero-initialize an array with as many rows as we have instances and as many cols as we have classes (num_classes)
    result = np.zeros((len(vector), num_classes), dtype='int32')
    # now pick in each row the corresponding col (given by vector) and set it to 1
    result[np.arange(len(vector)), vector] = 1 
    return result


# In[4]:

######--------- TEST Start----------------------------------################
# for function: convertToOneHot
'''
vector = [2,3] # is actually a list
convertToOneHot(vector, num_classes=10)
'''
####----------Test End----------------------------------------##########
########################################################################


# ### my_norm

# In[5]:

###################################################
# function for normalization over mini-batch: my_norm(Ylogits, scope)
# input: Ylogits (logits or activaions of a minibatch), scope (for naming)
# output: normalized input (substract from each activation the mean-of-batch and divide by sd-of-batch)
###################################################
def my_norm(Ylogits, scope):
    with tf.variable_scope(scope) as v_scope:
        mean, variance = tf.nn.moments(Ylogits, [0])
        m = mean
        v = variance
        bnepsilon = 1e-5 #A small float number to avoid dividing by 0
        Y_norm = tf.divide(tf.subtract(Ylogits, m),tf.sqrt(tf.add(v,bnepsilon)))
        return Y_norm


# In[6]:

######--------- TEST Start----------------------------------################
# for function: my_norm
'''
##test of the function my_norm:
inp = tf.placeholder(tf.float32, shape=[None, 8], name='input')
out_norm = my_norm(inp, scope="normalize")
# test 1: zero valued input:
res0 = out_norm.eval(session=tf.Session(), feed_dict={inp: np.zeros((4, 8))})
print(res0)
# test 2:  with non zero values
tmp = np.arange(32).reshape(4,8)
# check against direct calculation with numpy:
res = out_norm.eval(session=tf.Session(), feed_dict={inp: tmp})
check = (tmp-np.mean(tmp,axis=0))/np.sqrt(np.var(tmp,axis=0)+1e-5)
print(res - check)
'''
####----------Test End----------------------------------------##########
########################################################################


# ### my_fc_bn

# In[7]:

###################################################
# a simple batch-normalization fct:  my_fc_bn(Ylogits, offset, scope)
# input: Ylogits (logits or activaions of a minibatch), offset (for backlearnable shift), scope (for naming) 
# output: batch-normalized input
# a simple version fo batch-normalization only for fcNN during training w/o moving average
# this is a 2-step procedur, 1. step is a normalization over a minibatch (centering and scaling) 
# 2. step allows a potential learning back (via scale and offset)
# see also https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
# following M. Görner, we only profide a offset but no scale since we use (linear) ReLu 
###################################################
def my_fc_bn(Ylogits, offset, scope):
    with tf.variable_scope(scope) as v_scope:
        mean, variance = tf.nn.moments(Ylogits, [0])
        m = mean
        v = variance
        bnepsilon = 1e-8 #A small float number to avoid dividing by 0
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn


# In[8]:

######--------- TEST Start----------------------------------################
# for function: my_fc_bn
'''
Nb = 2
N1 = 4
N2 = 8
inp = tf.placeholder(tf.float32, shape=[None,N1, N2], name='input')
offs = tf.placeholder(tf.float32, shape=[N1,N2], name='offset')
bn_out = my_fc_bn(inp,offs, "BN")

# test 1: zero input
res0 = bn_out.eval(session=tf.Session(), feed_dict={inp: np.zeros((Nb,N1,N2)),offs:np.zeros((N1,N2))})
print("with input zero, the bn is:")
print(res0)
# test 2: non zero input
tmp = np.ones((Nb,N1,N2))   # initialize input array with ones (a batch of Nb units, each of size N1xN2)
tmp[:,::2,:]=2*tmp[:,::2,:] # for demonstration purpose: set every other row to 2
tmp[1] = 2*tmp[0]           # then set the second batch unit to be twice the first one
print("with the following non zero input:")
print(tmp)            

# tf batch norm
res = bn_out.eval(session=tf.Session(), feed_dict={inp: tmp,offs:np.zeros((N1,N2))})
# numpy batch norm
check = (tmp-np.mean(tmp,axis=0))/np.sqrt(np.var(tmp,axis=0)+1e-8)
# print the difference
print("the difference in bn between tf and np is:")
print(res - check)
# remark: there seems to be slight differences in the sqrt calculation of tf and np 
# which introduce a difference of O(1e-8) between the two results, also reported on in:
# https://stackoverflow.com/questions/45410644/tensorflow-vs-numpy-math-functions
'''
####----------Test End----------------------------------------##########
########################################################################

# ### my_rescale

# In[7b]:

###################################################
# a simple rescale bn fct:  my_rescale(Ylogits, offset, scope)
# input: Ylogits (logits or activaions of a minibatch), 
# offset (for backlearnable shift), scope (for naming) 
# output: rescaled minbatch
# this is only the 2. step of bn, 1. step is a normalization over a minibatch (centering and scaling) 
# this 2. step allows a potential learning back (via scale and offset)
# see also https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization
# following M. Görner, we only have offset in batchnorm but no scale gamma
# following Martin Görner no scales in bn is needed since we use ReLu as non-linear activation
###################################################
def my_rescale(Ylogits, offset, scope):
    with tf.variable_scope(scope) as v_scope:
        m = 0  # we alreade have standardized before
        v = 1
        bnepsilon = 1e-8 #A small float number to avoid dividing by 0
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn


    ######--------- TEST Start----------------------------------################
# for function: my_rescale
'''
Nb = 2
N1 = 4
N2 = 8
inp = tf.placeholder(tf.float32, shape=[None,N1, N2], name='input')
offs = tf.placeholder(tf.float32, shape=[N1,N2], name='offset')
bn_out = my_rescale(inp,offs, "BN")

# test 1: zero input
res0 = bn_out.eval(session=tf.Session(), feed_dict={inp: np.zeros((Nb,N1,N2)),offs:np.zeros((N1,N2))})
print("with input zero, the bn is:")
print(res0)
# test 2: non zero input
tmp = np.ones((Nb,N1,N2))   # initialize input array with ones (a batch of Nb units, each of size N1xN2)
tmp[:,::2,:]=2*tmp[:,::2,:] # for demonstration purpose: set every other row to 2
tmp[1] = 2*tmp[0]           # then set the second batch unit to be twice the first one
print("with the following non zero input:")
print(tmp)            

# tf batch norm
res = bn_out.eval(session=tf.Session(), feed_dict={inp: tmp,offs:np.zeros((N1,N2))})
# numpy batch norm
check = (tmp-0)/np.sqrt(1+1e-8)
# print the difference
print("the difference in bn between tf and np is:")
print(res - check)
# remark: there seems to be slight differences in the sqrt calculation of tf and np 
# which introduce a difference of O(1e-8) between the two results, also reported on in:
# https://stackoverflow.com/questions/45410644/tensorflow-vs-numpy-math-functions
'''
####----------Test End----------------------------------------##########
########################################################################


# ### my_noise

# In[9]:

##############################################################################
# function to introduce noise to layer: my_noise(input_layer, sd, scope)
# input: input_layer (mini-batch of activations), sd (stadev of of added N(0,sd), scope (for naming)
# output: input-batch plus added normal noise
# a simple batch-normalization fct:  my_fc_bn(Ylogits, offset, scope)
# input: Ylogits (logits or activaions of a minibatch), offset (for backlearnable shift), scope (for naming) 
# output: batch-normalized input
# taken from https://stackoverflow.com/questions/41174769/additive-gaussian-noise-in-tensorflow
#####################################################################################
def my_noise(input_layer, sd, scope):
    with tf.variable_scope(scope) as v_scope:
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=sd, dtype=tf.float32) 
        return input_layer + noise


# In[10]:

######--------- TEST Start----------------------------------################
# for function: my_noise
'''
inp = tf.placeholder(tf.float32, shape=[None, 8], name='input')
noise = my_noise(inp, sd=0.0, scope="noise_adding")  # for sd=0.0 we get back unchanged inp (use float numbers)
noise.eval(session=tf.Session(), feed_dict={inp: np.zeros((4, 8))})
'''
####----------Test End----------------------------------------##########
########################################################################


# ### intialize functions

# In[11]:

##############################################################################
# initialize weight matrices for encoder & decoder: 
# init_weights(layers, layer_level, sd, renorm, decoder, scope)
# input:
# layers (holding number number of nodes in each layer of the fc NN)
# layer_level (defines the layer from which we start)
# sd (holds the standard deviation of the normal N(0,sd) which we use for intitialization)
# renorm (number by which 1 is divided to intialize offsets in bn)
# decoder (decoder=0 to initialize weights for the encoder, =1 to initialize decoder), scope (for naming)
# scope (for naming)
# output: W (weight matrices for linear trafo), B (offsets in batchnorm)
# we use no biases in linear trafo since we do batchnorm (following Martin Görner)
# Bs are used as offset in batchnorm (no scales in bn since we use ReLu)
##################################################################################
def init_weights(layers, layer_level, sd, renorm, decoder, scope):
    with tf.variable_scope(scope) as v_scope:
        # shape of W and B needs to be different for encoder or decoder: 
        # if decoder=0 a=layer_level-1 and b=layer_level, if decoder=1, a=layer_level and b=layer_level-1
        a = layer_level - 1*(1-decoder) 
        b = layer_level - 1*decoder
        W = tf.Variable(tf.truncated_normal([layers[a], layers[b]], stddev=sd), name="weights_lt")  
        B = tf.Variable(tf.ones(layers[b])/renorm, name="offset_bn")
        return W,B



# In[12]:

######--------- TEST Start----------------------------------################
# test for function: init_weights
'''
tf.reset_default_graph()
L0 = 5  # first layer holds input 
L1 = 10  # last layer holds activations before softmax prediction of labels

# layers is a list holding the size of all layers in the fc NN
layers=[L0, L1]

my_sd = 0.1  # sd for initialization of W with N(0,sd)
fan = 10     # for dividing the 1-initialized B
# learnable variables in encoder (decoder=0):
W1, B1 = init_weights(layers, layer_level=1, sd=my_sd, renorm=fan, decoder=0, scope="var_enc_1")

# learnable variables in decoder:
WD1, BD1 = init_weights(layers, layer_level=1, sd=my_sd, renorm=fan, decoder=1, scope="var_dec_1")

## test dimensions
init_op = tf.global_variables_initializer() 

# run the graph
sess = tf.Session()
sess.run(init_op) #initialization on the concrete realization of the graph
W1, B1, WD1, BD1  = sess.run(fetches=(W1, B1, WD1, BD1)) 

print("shape W1:",np.shape(W1))
print("shape B1:",np.shape(B1))
print("shape WD1:",np.shape(WD1))
print("WD1:",WD1)
print("shape BD1:",np.shape(BD1))
print("BD1:",BD1)
'''
####----------Test End----------------------------------------##########
########################################################################

# In[12b]:

##############################################################################
# initialize weight matrices for encoder & decoder: 
# init_weights_v2(layers, layer_level, min_w, max_w, decoder, scope)
# input:
# layers (holding number number of nodes in each layer of the fc NN)
# layer_level (defines the layer from which we start)
# min_w (holds the min value of the random_uniform distribution)
# max_w (holds the min value of the random_uniform distribution)
# decoder (decoder=0 to initialize weights for the encoder, =1 to initialize decoder), scope (for naming)
# scope (for naming)
# output: W (weight matrices for linear trafo), B (offsets in batchnorm)
# only offset Bs are used as offset in batchnorm (no scales in bn since we use ReLu)
##################################################################################
def init_weights_v2(layers, layer_level, min_w, max_w, decoder, scope):
    with tf.variable_scope(scope) as v_scope:
        # shape of W and B needs to be different for encoder or decoder: 
        # if decoder=0 a=layer_level-1 and b=layer_level, if decoder=1, a=layer_level and b=layer_level-1
        a = layer_level - 1*(1-decoder) 
        b = layer_level - 1*decoder
        W = tf.Variable(tf.random_uniform([layers[a], layers[b]], minval=min_w, maxval=max_w), name="weights_lt") 
        B = tf.Variable(tf.zeros(layers[b]), name="offset_bn")
        return W,B

################################################################################
# ### encoder_prop function

# In[13]:

#############################################
# clean or noisy encoder construction  - general fcn encoder_prop(h_in, W, B, noise, scope)
# input: h_in: signal from below, W: for the linear trafo, B: offset in bn, noise: None (in clean encoder) or sd_noise-value
# output: hn_lt, hn_norm, hn_noise, hn_bn, hn_nlt
###################################################
def encoder_prop(h_in, W, B, noise, scope):
    with tf.variable_scope(scope) as v_scope:
        # Biases in linear trafo are actually not useful when using BN (see Martin Görner)
        hn_lt = tf.matmul(h_in, W)  
        hn_norm = my_norm(Ylogits=hn_lt, scope="normalize") 
        if noise is not None:
            hn_noise = my_noise(hn_norm, noise, "noise_adding")
        else:
            hn_noise = hn_norm
        # scale in bn is not useful in front of ReLu
        #hn_bn = my_fc_bn(Ylogits=hn_noise, offset=B, scope="bn")
        hn_bn = my_rescale(Ylogits=hn_noise, offset=B, scope="bn")

        hn_nlt = tf.nn.relu(hn_bn)
        return hn_lt, hn_norm, hn_noise, hn_bn, hn_nlt



# In[15]:

######--------- TEST Start----------------------------------################
# test for function: encoder_prop
'''
# define input dimension
n_batch = 2 # batch size: # images in mini-batch
n_neurons = 5 # layer size

L0 = 5  # first layer holds input 
L1 = 10  
# layers is a list holding the size of all layers in the fc NN
layers=[L0, L1]

sd_noise = 0.2
my_sd = 0.1  # sd for initialization of W with N(0,sd)
fan = 10     # for dividing the 1-initialized B

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[n_batch, n_neurons])

# learnable variables in encoder (decoder=0):
W1, B1 = init_weights(layers, layer_level=1, sd=my_sd, renorm=fan, decoder=0, scope="var_enc_1")

_, _, hn_1_noise, _, hn_1_nlt = encoder_prop(x, W=W1, B=B1, noise=sd_noise, scope="n_encoder_1")
_, h_1_norm, _, _, h_1_nlt = encoder_prop( x , W=W1, B=B1, noise=None, scope="cl_encoder_1")

init_op = tf.global_variables_initializer()

# define input 
my_input = np.arange(n_batch*n_neurons, dtype='float32').reshape((n_batch,n_neurons))  

print("x.shape:",x.shape)  
print("my.input.shape:",my_input.shape)  
print("my.input:")
print(my_input) 

sess = tf.Session()
sess.run(init_op)

hn_1_noiseRes, hn_1_nltRes, h_1_normRes, h_1_nltRes = sess.run(fetches=(hn_1_noise, hn_1_nlt, h_1_norm, h_1_nlt), 
                                                  feed_dict={x: my_input}) 

print("hn_1_noiseRes.shape:",hn_1_noiseRes.shape)
print("hn_1_noiseRes:")
print(hn_1_noiseRes)
print("hn_1_nltRes.shape:",hn_1_nltRes.shape)
print("hn_1_nltRes:")
print(hn_1_nltRes)

print("h_1_normRes.shape:",h_1_normRes.shape)
print("h_1_normRes:")
print(hn_1_noiseRes)
print("h_1_nltRes.shape:",h_1_nltRes.shape)
print("h_1_nltRes:")
print(h_1_nltRes)
'''
####----------Test End----------------------------------------##########
########################################################################


# In[16]:

#############################################
# clean simple encoder construction  - general fcn simple_encoder_prop(h_in, W, B, noise, scope)
# input: h_in: signal from below, W: for the linear trafo, B: offset in bn, noise: None (in clean encoder) 
# output: h_lt, h_bn, h_nlt
###################################################
def simple_encoder_prop(h_in, W, B, noise, scope):
    with tf.variable_scope(scope) as v_scope:
        h_lt = tf.matmul(h_in, W)    # Biases in linear trafo are actually not useful when using BN
        h_bn = my_fc_bn(Ylogits=h_lt, offset=B, scope="bn")
        h_nlt = tf.nn.relu(h_bn)
        return h_lt, h_bn, h_nlt

# fill in test


# In[17]:

#############################################
# clean simple encoder construction  - general fcn simple_encoder_prop2(h_in, W, B, noise, scope):
# input: h_in: signal from below, W: for the linear trafo, B: offset in bn, noise: None (in clean encoder) 
# output: h_lt, h_nlt
###################################################
def simple_encoder_prop2(h_in, W, B, noise, scope):
    with tf.variable_scope(scope) as v_scope:
        h_lt = tf.add(tf.matmul(h_in, W),B) # with bias to use w/o bn  
        h_nlt = tf.nn.relu(h_lt)
        return h_lt, h_nlt
# fill in test 


# ## Combiner 

# In[18]:

'''
## do for demonstration of the procedure the combining of lateral and vertical layer step-wise
## example for tf.multiply with broadcasting
# can be run independent of all other cells

d1 = 2 # min-batch size
d2 = 5 # layer size (number of neurons, same for lat and vert input) index i
d3 = 4 # number of weights per neuron, index j

# generate matrix Z using ones, lateral, vertical and lateral*vertical
lateral = tf.constant(np.arange(1, (d1*d2)/2+1, dtype=np.float32),shape=[d1, d2])
vertical = tf.constant(np.arange(0, (d1*d2)/2, dtype=np.float32),shape=[d1, d2])
z = tf.reshape(lateral,[d1,d2,1])
u = tf.reshape(vertical,[d1,d2,1])
zu = tf.reshape(tf.multiply(lateral,vertical),[d1,d2,1])
# construct tensor for combiner operation (see below for explanation)
Z = tf.concat([tf.ones([d1,d2,1]),z,u,zu],axis = 2)

wsig = tf.constant(np.arange(1, d2+1, dtype=np.float32))#make sure to define is as a 1 dim tensor such that the tf.multiply works later on
W0 = tf.constant(np.arange(1, d2*d3+1, dtype=np.float32),shape=[d2, d3])
W1 = tf.constant(np.arange(0, d2*d3, dtype=np.float32),shape=[d2, d3])
#Z = tf.constant(np.arange(1, (d1*d2*d3)/2+1, dtype=np.float32),shape=[d1, d2, d3])

logit0 = tf.multiply(W0,Z) # uses broadcasting of tf.multiply
sum0 = tf.reduce_sum(logit0,axis=2) # sum over the d3=4 weight types
logit1 = tf.multiply(W1,Z)
sum1 = tf.reduce_sum(logit1,axis=2)

# to multiply with wsig, must reshape it to correspond to th
#g =  m0 +  tf.multiply(tf.reshape(wsig,shape = tf.shape(m1)[1:2]),tf.sigmoid(m1))## only if wsig is not defined as a 1d tensor
g =  sum0 +  tf.multiply(wsig,tf.sigmoid(sum1))

lat = lateral.eval(session=tf.Session())
ver = vertical.eval(session=tf.Session())
Zres = Z.eval(session=tf.Session())
print("input mini-batch from lateral, lat=")
print(lat)
print("input mini-batch from vertical, ver=")
print(ver)
print("We do for each image in mini-batch the combiner stuff")
print("for this we construct a tensor Z")
print("Z is a 4-dim tensor, with first dim is #images in mini-batch")
print("and for each iamge in the minibatch the 3-dim structure has 3 columns:")
print("ones (biases), lateral_input, vertical_input, lat*vert, Z=")
print(Zres)
W0res = W0.eval(session=tf.Session())
W1res = W1.eval(session=tf.Session())
logit0res = logit0.eval(session=tf.Session())
logit1res = logit1.eval(session=tf.Session())
sum0res = sum0.eval(session=tf.Session())
sum1res = sum1.eval(session=tf.Session())
gres = g.eval(session=tf.Session())
print("W0 holds weights for first weighted sum of 1, lat, vert, lat*vert:")
print("W0=",W0res)
print("W1 holds weights for 2. weighted sum (within sigmoid) of 1, lat, vert, lat*vert:")
print("W1=",W1res)
print("get logits by element-wise multiplication of weight-matrices with each image-element of Z")
print("logit0=",logit0res)
print("logit1=",logit1res)
print("get k-th row-element of the combined layer by summing up k-th row of corresponding logit matrix")
print("sum0=",sum0res)
print("sum1=",sum1res)
print("result of combining lateral and vertical input for each image in mini-batch g=")
print(gres)
'''


# In[19]:

#####################################################
# vanilla cominator as formula (16) in  https://arxiv.org/abs/1511.06430
# combiner of lat and vertical input - by fcn my_comb_vanilla(lateral, vertical, W0, W1, wsig, scope)
# input: lateral, vertical, W0, W1, wsig (input from lat and ver and variables for combinor function)
# output: g - combined layer which has the same dimensions as lateral and vertical
# or extended output for testing purposes
###################################################
def my_comb_vanilla(lateral, vertical, W0, W1, wsig, scope):
    with tf.variable_scope(scope) as v_scope:
        # generate tensor Z. Z has d1 units (d1 is the batch size). 
        # each unit is a matrix with d2 rows (d2 is size of layer) and 4 columns (4: z, u, zu, bias).
        # the columns correspond to: ones (bias), lateral layer, vertical layer and lateral*vertical
        with tf.name_scope("prep_comb"):
            d1 = tf.shape(lateral)[0]  # mini-batch-size
            d2 = tf.shape(lateral)[1]  # layer-size (same for lateral or vertical)
            with tf.name_scope("bias_prep"):
                bias = tf.ones([d1,d2,1])         # for each batch and neuron in layer one bias 1
            with tf.name_scope("z_lat_prep"):
                z = tf.reshape(lateral,[d1,d2,1]) # for each batch and neuron in layer one z-value
            with tf.name_scope("u_vert_prep"):
                u = tf.reshape(vertical,[d1,d2,1]) # for each batch and neuron in layer one u-value
            with tf.name_scope("zu_prod_prep"):
                zu = tf.reshape(tf.multiply(lateral,vertical),[d1,d2,1]) # for each batch and neuron in layer one zu-value
            with tf.name_scope("1_z_u_zu"):
                Z = tf.concat([bias,z,u,zu],axis = 2)
        with tf.name_scope("do_comb"):
            logit0 = tf.multiply(W0,Z) # uses broadcasting of tf.multiply, do element-wise mult for each element in batch
            sum0 = tf.reduce_sum(logit0,axis=2) # weighted sum - sum over the 4 contribution of bisas, z, u, zu
            logit1 = tf.multiply(W1,Z)
            sum1 = tf.reduce_sum(logit1,axis=2)
            # to multiply with wsig, must reshape it to correspond to th
            #g =  m0 +  tf.multiply(tf.reshape(wsig,shape = tf.shape(m1)[1:2]),tf.sigmoid(m1))## only if wsig is not defined as a 1d tensor
            g =  sum0 +  tf.multiply(wsig,tf.sigmoid(sum1))
            #return bias, z, u, zu, Z, g  # only for testing the function
            return g


# In[20]:

######--------- TEST Start----------------------------------################
# test for function: my_comb_vanilla
'''
# above we need the return with all elements - out-comment "return g"
# test of function my_comb_vanilla
# construct the graph:
tf.reset_default_graph()
# define input dimension
n1 = 2 # batch size: # images in mini-batch
n2 = 5 # layer size
inp1 = tf.placeholder(tf.float32, shape=[n1, n2], name='input')
inp2 = tf.placeholder(tf.float32, shape=[n1, n2], name='input')

# initialize weight matrices
# define auxiliary tensors for convenience
li1 = tf.range(n2,dtype=tf.float32)
ze1 = tf.zeros(n2)
on1 = tf.ones(n2)
ze = tf.zeros([n2,1])
on = tf.ones([n2,1])
li = tf.reshape(li1,[n2,1])

# construct weight matrices by concatinating
W0init = tf.concat([ze,on,ze,ze],1)
W1init = tf.concat([ze,on,ze,ze],1)
# other examples for initialization values:
#W0init = tf.concat([li,on,ze,li],1)
#W1init = tf.concat([on,on,ze,li],1)

# define weight matirces as initialized variables
# define it as 1 dim tensor such that the tf.multiply works later on
W0_ = tf.Variable(W0init,name = 'W0_')
W1_ = tf.Variable(W1init,name = 'W1_')
wsig_ =  tf.Variable(ze1, name = 'wsig_')

# symbolic call to combinator
bias_res1, z_res1, u_res1, zu_res1, Z_res1, g_res1 = my_comb_vanilla(inp1, inp2, W0_, W1_, wsig_,"combiner")

init_op = tf.global_variables_initializer() 

# define input tensors
in2 = np.ones((n1,n2))
inlin = np.arange(n1*n2).reshape((n1,n2))/n1/n2

# run the graph
sess = tf.Session()
sess.run(init_op) #initialization on the concrete realization of the graph

bias_res, z_res, u_res, zu_res, Z_res, g_res = sess.run([bias_res1, z_res1, u_res1, zu_res1, Z_res1, g_res1], 
                                    feed_dict={inp1: inlin, inp2:in2}) #Evaluation result from fct my_combiner_w

W0res, W1res, wsigres = sess.run([W0_, W1_, wsig_], 
                                    feed_dict={inp1: inlin, inp2:in2}) #Evaluation result from fct my_combiner_w

print("inputs:")
print("lateral:")
print(inlin)
print("vertical:")
print(in2)
print("W0=",W0res)
print("W1=",W1res)
print("wsig=",wsigres)
print("====================")
print("the prep_comb tf result is:")
#print("bias:")
#print(bias_res)
#print("z:")
#print(z_res)
#print("u:")
#print(u_res)
#print("zu:")
#print(zu_res)
print("Z concat - bias, z, u, zu:")
print(Z_res)
print("the do_comb: only multiply elementwise W0 and Z (giving the only contribution to g since wsig is 0)")
print("sum over 4 contributions in one row to get the value of one neuron arranged as row-elments (we have 5 nerons per layer)")

print("combined:")
print(g_res)
# calculate with np
lat = inlin
ver = in2
sum0 =  W0res[:,0]*np.ones(np.shape(lat)) + W0res[:,1]*lat + W0res[:,2]*ver + W0res[:,3]*lat*ver
sum1 =  W1res[:,0]*np.ones(np.shape(lat)) + W1res[:,1]*lat + W1res[:,2]*ver + W1res[:,3]*lat*ver
gnp = sum0 + wsigres / (1+np.exp(-sum1))
print("====================")
print("the np result is:")
print(gnp)
'''


# In[21]:

#########################################################################
# initialize weight matrices for vanilla combiner: init_weights_combiner_vanilla(n2, scope)
# input: n2 (layer length, same for lateral and vertical input), sope (for naming)
# output: wsig_ (weight of simoid of 2. sum), W0_ (weights for 1. sum of bias, lat, vert, lat*vert), W1_ (for 2. sum)
# folloing formula (17) - lateral gets intially full weight and sigmoid too
############################################################################
'''
def init_weights_combiner_vanilla(n2, scope):
    with tf.variable_scope(scope) as v_scope:
        # define auxiliary tensors for convenience
        ze1 = tf.zeros(n2)
        on1 = tf.ones(n2)  # used for lateral weight intitializing
        ze = tf.zeros([n2,1])
        on = tf.ones([n2,1])

        # construct weight matrices by concatinating
        W0init = tf.concat([ze,on,ze,ze],1)
        W1init = tf.concat([ze,on,ze,ze],1)

        # define weight matirces as initialized variables
        wsig_ =  tf.Variable(ze1, name = 'wsig_')#make sure to define is as a 1 dim tensor such that the tf.multiply works later on
        W0_ = tf.Variable(W0init, name = 'W0_')
        W1_ = tf.Variable(W1init, name = 'W1_')
        return W0_,W1_,wsig_
'''

# In[22]:

######--------- TEST Start----------------------------------################
# missing test for function init_weights_combiner_vanilla
####----------Test End----------------------------------------##########
########################################################################


# ### decoder_prop function

# In[23]:

#############################################
# decoder construction  - general fcn decoder_prop(h_lat, h_ver, W0comb, W1comb, wsigcomb, W, B, scope)
# input: for combiner: h_lat, h_ver, W0comb, W1comb, wsigcomb, for linear-trafo: W, B, scope (naming for graf)
# output: d_c (needed in reconstruction cost) , d_lt, d_norm
###################################################
def decoder_prop(h_lat, h_ver, W0comb, W1comb, wsigcomb, W, B, scope):
    # h_lat, h_ver:lateral and vertical inputs. 
    # W (weights for linear trafo), B (bias for linear trafo - in decoder we have no bn) 
    # W0comb,W1comb,wsigcomb: combiner weights
    with tf.variable_scope(scope) as v_scope:
        d_c = my_comb_vanilla(h_lat, h_ver, W0comb, W1comb, wsigcomb, "combiner")
        d_lt = tf.add(tf.matmul(d_c, W) ,B)     # linear tranformation 
        d_norm = my_norm(Ylogits=d_lt, scope="normalize") # normalization
        return d_c, d_lt, d_norm


# In[24]:

######--------- TEST Start----------------------------------################
# missing test for function: decoder_prop

####----------Test End----------------------------------------##########
########################################################################


# ### reconst_loss function

# In[25]:

'''
#############################################
# loss unsupervised per layer  - general fcn reconst_loss(encoder_clean, decoder, scope):
# formula (18): normalize dedcoder using *encoder’s* sample mean and standard deviation statistics
# input: encoder_clean, decoder (same shape than encoder_clean), scope
# output: loss_reconst (squared Euclidean distance between clean encoder layer and corresponding decoder layer)
###################################################
def reconst_loss(encoder_clean, decoder, scope):
    with tf.name_scope("squared_dist"):
        # normalize decoder using *encoder’s* sample mean and variance
        mean, variance = tf.nn.moments(encoder_clean, axes=[0])
        m = mean
        v = variance
        bnepsilon = 1e-5 #A small float number to avoid dividing by 0
        decoder_norm2 = tf.divide(tf.subtract(decoder, m),tf.sqrt(tf.add(v,bnepsilon)))
        loss_reconst = tf.reduce_mean(tf.pow(encoder_clean - decoder_norm2, 2))
        return loss_reconst
'''


# In[ ]:

######--------- TEST Start----------------------------------################
# missing test for function reconst_loss
####----------Test End----------------------------------------##########
########################################################################

