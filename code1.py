# this code runs a multilayer perceptron algorithm with 1 hidden layer containing 2 nodes.
import q5
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class out_neuron:
    inputs = np.array([1,0,0])
    inp_weights = np.array([1,1,1])
    activation = 0
    output = 0
    loc_grad = 0
    def comp_act(self):
        return np.dot(self.inputs,self.inp_weights)
    def comp_out(self):
        return (1/(1+(math.e)**(-0.01*self.activation)))
    def comp_loc_grad(self,des):
        #print(self.loc_grad)
        return (des - self.output)*self.output*(1-self.output)
        #print(self.loc_grad)

class hid_neuron:
    inputs = [1,0,0,0,0]
    inp_weights = [1,1,1,1,1]
    out_weights = [1,1,1]
    activation = 0
    output = 0
    loc_grad = 0
    def comp_act(self):
        return np.dot(self.inputs,self.inp_weights)
    def comp_out(self):
        return (1/(1+(math.e)**(-0.01*self.activation)))



def forward_propagate(n,inputs):
    for i in range(5):
        n[0].inputs[i] = inputs[i]
        n[1].inputs[i] = inputs[i]
    n[0].activation = n[0].comp_act()
    n[0].output = n[0].comp_out()
    n[1].activation = n[1].comp_act()
    n[1].output = n[1].comp_out()
    n[2].inputs = np.array([1,n[0].output,n[1].output])
    n[2].activation = n[2].comp_act()
    n[2].output = n[2].comp_out()
    n[3].inputs = np.array([1,n[0].output,n[1].output])
    n[3].activation = n[3].comp_act()
    n[3].output = n[3].comp_out()
    n[4].inputs = np.array([1,n[0].output,n[1].output])
    n[4].activation = n[4].comp_act()
    n[4].output = n[4].comp_out()
    #print(n[0].activation,n[1].activation,n[2].activation,n[3].activation,n[4].activation)
    return n

def back_propagate(n,des):
    n[2].loc_grad = n[2].comp_loc_grad(des[0])
    n[3].loc_grad = n[3].comp_loc_grad(des[1])
    n[4].loc_grad = n[4].comp_loc_grad(des[2])
    #print(n[0].loc_grad,n[1].loc_grad,n[2].loc_grad,n[3].loc_grad,n[4].loc_grad)
    temp = n[0].out_weights[0]*(n[2].loc_grad) + n[0].out_weights[1]*(n[3].loc_grad) + n[0].out_weights[2]*(n[4].loc_grad)
    n[0].loc_grad = temp*(n[0].output)*(1 - n[0].output)
    temp = n[1].out_weights[0]*(n[2].loc_grad) + n[1].out_weights[1]*(n[3].loc_grad) + n[1].out_weights[2]*(n[4].loc_grad)
    n[1].loc_grad = temp*(n[1].output)*(1 - n[1].output)
    return n
def update_weights(n):
    #learn = 1
    for i in range(5):
        n[0].inp_weights[i] = n[0].inp_weights[i] + (n[0].loc_grad * n[0].inputs[i])
        n[1].inp_weights[i] = n[1].inp_weights[i] + (n[1].loc_grad * n[1].inputs[i])
    for j in range(3):
        n[2].inp_weights[j] = n[2].inp_weights[j] + (n[2].loc_grad * n[2].inputs[j])
        n[3].inp_weights[j] = n[3].inp_weights[j] + (n[3].loc_grad * n[3].inputs[j])
        n[4].inp_weights[j] = n[4].inp_weights[j] + (n[4].loc_grad * n[4].inputs[j])
        #print(n[3].inp_weights)
    #print(n[0].loc_grad)
    n[0].out_weights[0] = n[2].inp_weights[1]
    n[0].out_weights[1] = n[3].inp_weights[1]
    n[0].out_weights[2] = n[4].inp_weights[1]
    n[1].out_weights[0] = n[2].inp_weights[2]
    n[1].out_weights[1] = n[3].inp_weights[2]
    n[1].out_weights[2] = n[4].inp_weights[2]
    #print(n[0].inp_weights)
    return n

def main_alg(n):
    ar = np.random.permutation(150)
    for i in ar:
        #print(n[0].out_weights)
        temp_n = forward_propagate(n,q5.data_lst[i])
        for j in range(5):
           n[j] = temp_n[j]
        temp_n = back_propagate(n,q5.desired[i])
        print(q5.desired[i])
        for j in range(5):
           n[j] = temp_n[j]
        temp_n = update_weights(n)
        for j in range(5):
           n[j] = temp_n[j]
    return n
def predict_error(n):
    ar = np.random.permutation(150)
    for i in ar:
        forward_propagate(n,q5.data_lst[i])
        prediction = np.argmax([n[2].output,n[3].output,n[4].output])
        print(prediction)


n1 = hid_neuron()
n2 = hid_neuron()
n3 = out_neuron()
n4 = out_neuron()
n5 = out_neuron()
n = [n1,n2,n3,n4,n5]
n = main_alg(n)
#print(n[3].inp_weights)
predict_error(n)
