import q5
import numpy as np
import matplotlib.pyplot as plt
import random
import math

#5X2 matrix for input to hidden connection and 3X3 for hidden to output multilayer

#a = np.array([[1,2],[1,2],[1,2],[1,2],[1,2]],dtype = 'float')
#b = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype = 'float')
a = np.random.rand(5,2)
b = np.random.rand(3,3)

err = []
for epoch in range(1000):
    order = np.random.permutation(150)
    for i in order:
        temp_inp = np.array(q5.data_lst[i])
        #for k in range(1,5):
        #    temp_inp[k] = 1/(1 + np.exp(-temp_inp[k]))
        #this is for the forward propagation
        hidden_output = np.dot(temp_inp,a)#1X2 vector
        #print(temp_inp)
        #print(hidden_output)
        #print(hidden_output)
        hidden_activation = np.array([1,0,0],dtype = 'float')
        for j in range(1,3):
            hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X3 vector(first element is the bias), these serve as inputs for the output layer
        #print(hidden_activation)
        output = np.dot(hidden_activation,b)
        final_output = np.array([0,0,0],dtype = 'float')
        #print(final_output)
        for j in range(3):
            final_output[j] = 1/(1 + np.exp(-output[j])) #1X3 vector, this is the final_output

        #this completes the forward propagation, now we implement the backward propagation
        des = np.array(q5.desired[i],dtype = 'float')
        #first we compute the local gradient of the output nodes
        output_loc_grad = np.multiply(np.multiply((des - final_output),final_output),(np.array([1,1,1])-final_output))#1X3 vector
        #print(output_loc_grad)
        #now to calculate the local gradient of the neurons in the hidden layer
        bT = np.transpose(b)
        #temp_act = np.array([hidden_activation[1],hidden_activation[2]])#excludes the bias from the neurons in the hidden layer
        hidden_loc_grad = np.multiply(np.multiply(np.dot(output_loc_grad,bT),hidden_activation),(np.array([1,1,1]) - hidden_activation))#1X3 vector
        hidden_loc_grad1 = np.array([hidden_loc_grad[1],hidden_loc_grad[2]])#this is a 1X2 vector which removes the bias neuron
        #print(hidden_loc_grad)
        # now to update the weights after every Iterations
        learn = 5
        #first we update the matrix 'a'
        for r in range(5):
            for c in range(2):
                a[r][c] = a[r][c] + learn*hidden_loc_grad1[c]*temp_inp[r]

        # now we update the matrix 'b'
        for r in range(3):
            for c in range(3):
                b[r][c] = b[r][c] + learn*output_loc_grad[c]*hidden_activation[r]

#this is to check the performance using the final weights for prediction
print(a)
print(b)
ar = np.random.permutation(150)
error = 0
for i in ar:
    temp_inp = np.array(q5.data_lst[i])
    des = np.array(q5.desired[i])
    #for k in range(1,5):
    #    temp_inp[k] = 1/(1 + np.exp(-temp_inp[k]))
    #this is for the forward propagation
    hidden_output = np.dot(temp_inp,a)#1X2 vector
    #print(temp_inp)
    #print(hidden_output)
    #print(hidden_output)
    hidden_activation = np.array([1,0,0],dtype = 'float')
    for j in range(1,3):
        hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X3 vector(first element is the bias), these serve as inputs for the output layer
    #print(hidden_activation)
    output = np.dot(hidden_activation,b)
    final_output = np.array([0,0,0],dtype = 'float')
    #print(final_output)
    for j in range(3):
        final_output[j] = 1/(1 + np.exp(-output[j])) #1X3 vector, this is the final_output
    #print("Final Output: ",final_output)
    #print("Max Final Output: ",np.argmax(final_output))
    #print("Desired Output: ",des)
    #print("Desired Type: ",np.argmax(des))
    if np.argmax(des) != np.argmax(final_output):
        error = error + 1
    #err.append(error)
#print("Error: ",error)
print(error)
