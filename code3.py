import q5
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import matplotlib.pyplot as plt
#this code has 8 neurons in the hidden layer
#5X8 matrix for input to hidden connection and 9X3 for hidden to output multilayer

#a = np.array([[1,2],[1,2],[1,2],[1,2],[1,2]],dtype = 'float')
#b = np.array([[1,1,1],[1,1,1],[1,1,1]], dtype = 'float')
a = np.random.rand(5,8)
b = np.random.rand(9,3)
a_list = [a,a]# this stores the list of a matrices
b_list = [b,b]# this stores the list of b matrices
err = []
count = 0
full_order = np.random.permutation(150)
training = []
testing = []
for f in range(100):# using 100 data points for training
    training.append(full_order[f])
for g in range(100,150):# using 50 data points for testing
    testing.append(full_order[g])
for epoch in range(1000):
    for i in training:
        count = count + 1
        temp_inp = np.array(q5.data_lst[i])
        #for k in range(1,5):
        #    temp_inp[k] = 1/(1 + np.exp(-temp_inp[k]))
        #this is for the forward propagation
        hidden_output = np.dot(temp_inp,a)#1X8 vector
        #print(temp_inp)
        #print(hidden_output)
        #print(hidden_output)
        hidden_activation = np.array([1,0,0,0,0,0,0,0,0],dtype = 'float')
        for j in range(1,9):
            hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X9 vector(first element is the bias), these serve as inputs for the output layer
        #print(hidden_activation)
        output = np.dot(hidden_activation,b)#this is a 1X3 matrix
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
        hidden_loc_grad = np.multiply(np.multiply(np.dot(output_loc_grad,bT),hidden_activation),(np.array([1,1,1,1,1,1,1,1,1]) - hidden_activation))#1X9 vector
        hidden_loc_grad1 = np.array([hidden_loc_grad[1],hidden_loc_grad[2],hidden_loc_grad[3],hidden_loc_grad[4],hidden_loc_grad[5],hidden_loc_grad[6],hidden_loc_grad[7],hidden_loc_grad[8]])#this is a 1X8 vector which removes the bias neuron
        #print(hidden_loc_grad)
        # now to update the weights after every Iterations
        learn = 1
        alpha = 1
        #first we update the matrix 'a'
        for r in range(5):
            for c in range(8):
                a[r][c] = a[r][c] + ((a_list[count][r][c] - a_list[count][r][c])*alpha) + learn*hidden_loc_grad1[c]*temp_inp[r]# includes the momentum term

        # now we update the matrix 'b'
        for r in range(9):
            for c in range(3):
                b[r][c] = b[r][c] + ((b_list[count][r][c] - b_list[count][r][c])*alpha) + learn*output_loc_grad[c]*hidden_activation[r]#includes the momentum term
        a_list.append(a)
        b_list.append(b)
#this is for plotting the error trajectory
#print(a)
#print(b)
#ar = np.random.permutation(150)
    if epoch%50 == 0:
        error = 0
        for t in testing:
            temp_inp = np.array(q5.data_lst[t])
            des = np.array(q5.desired[t])
            hidden_output = np.dot(temp_inp,a)#1X8 vector
            #print(temp_inp)
            #print(hidden_output)
            #print(hidden_output)
            hidden_activation = np.array([1,0,0,0,0,0,0,0,0],dtype = 'float')
            for j in range(1,9):
                hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X9 vector(first element is the bias), these serve as inputs for the output layer
            #print(hidden_activation)
            output = np.dot(hidden_activation,b)#this is a 1X3 matrix
            final_output = np.array([0,0,0],dtype = 'float')
            #print(final_output)
            for j in range(3):
                final_output[j] = 1/(1 + np.exp(-output[j])) #1X3 vector, this is the final_output
            if np.argmax(des) != np.argmax(final_output):
                error = error + 1
        err.append(error)
            #err.append(error)
        #print("Error: ",error)
        #print(error)
#print(err)
#plot code
'''
error = 0
for t in testing:
    temp_inp = np.array(q5.data_lst[t])
    des = np.array(q5.desired[t])
    hidden_output = np.dot(temp_inp,a)#1X8 vector
    #print(temp_inp)
    #print(hidden_output)
    #print(hidden_output)
    hidden_activation = np.array([1,0,0,0,0,0,0,0,0],dtype = 'float')
    for j in range(1,9):
        hidden_activation[j] = 1/(1 + np.exp(-hidden_output[j-1]))#1X9 vector(first element is the bias), these serve as inputs for the output layer
    #print(hidden_activation)
    output = np.dot(hidden_activation,b)#this is a 1X3 matrix
    final_output = np.array([0,0,0],dtype = 'float')
    #print(final_output)
    for j in range(3):
        final_output[j] = 1/(1 + np.exp(-output[j])) #1X3 vector, this is the final_output
    if np.argmax(des) != np.argmax(final_output):
        error = error + 1
print(error)
'''
print(err[19])
x_list = []
for i in range(1, 1001):
    if i%50 == 0:
       x_list.append(i)
plt.scatter(x_list,err)
plt.plot(x_list,err)
plt.xlabel("Iterations")
plt.ylabel("%error")
plt.title("Error Trajectory")
plt.show()
