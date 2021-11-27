import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#dataset is from kaggle.com
#dataframe to work with
df = pd.read_csv(spotify.csv')
df.head()
# Refer pic-1 in output.dic

#drop irrelevant attributes
#df.drop(['track_id','acousticness','duration_ms','key','instrumentalness','time_signature'],axis=1)
#writing code for general neural network
df neuralnetwork(inp1,inp2,w1,w2,b):
   output = inp1*w1 + inp2*w2+b
   return normalise(output)
#writing code for normalising
def normalise(x):
    return 1/(1+np.exp(-x))
#replace sad with 0 and happy with 1
new_arr = []
for i in dif['mood']:
    if i=='Sad':
        new_arr.append(0)
    else:
        new_arr.append(1)
#print(new_arr)
df['result']=new_arr
ldata = len(df)
print(ldata)
#80% training 20% test -------- fixing length of training and test data set(not needed)
trainlen = int(ldata*0.8)
testlen = ldata-trainlen
i=0
#we are going to work with energy ,volence ,mood -----> x , y , output
energy = np.array(df['energy'])
valence=np.array(df['valence'])
output = np.array(df['result'])
print(energy,valence,output)
#refer pic-2 in output.doc

len(energy)
#refer pic-3 in output.doc

dt = []
#now we have to make the data in form of a[x,y,output]
for t in range(len(energy)):
    dt.append([energy[t],valence[t],output[t]])
dt
#refer pic-4 in output.doc

#Lets see the data now
dtl = len(dt)
for z in range(100):
    plt.axis([0,1,0,1])
    point = dt[z]
    color = 'r' #sadsongs
    if point[2] == 1:
        color='k'   #happysongs
    plt.scatter(point[0],point[1],c=color)
##refer pic-5 in output.doc

def normalise_derivative(n):
    return normalise(n)*(1-normalise(n))
#setting random weights
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
print(w1,w2)
#training the data
learning_rate = 0.7
costs=[]
for i in range(12000):
    pos = np.random.randint(len(dt))
    dat = dt[pos]
    input1 = dat[0]
    input2 = dat[1]
    out = dat[2]
    z = input1*w1+input2*w2+b
    prediction - neuralnetwork(input1,input2,w1,w2,b)
    cost = np.square(prediction-out)
    costs.append(cost)
    #derivatives -- - -- -- - -- -  dcost_pred = {d/d(pred){(pred-out)**2 similarly for all
    dcost_pred - 2*(prediction-out)
    dpred_z=normalise_derivative(z)
    dz_w1=input1
    dz_w2=input2
    dz_b=1
    #applying chain rule
    dcost_z = dcost_pred*dpred_z
    dost_w1=dcost_z*dz_w1
    dcost_w2=dcost_z*dz_w2
    dcost_b=dcost_z*dz_b
    #substracting the learning rate
    w1 = w1 - learning_rate*dcost_w1
    w2 = w2 - learning_rate*dcost_w2
    b = b - learning_rate*dcost_b
    #refer pic-6 in output.doc

    plt.plot(costs)
    #refer pic-7 in output.doc

    predicted_results=[]
    for i in range(len(dt)):
        pt = dt[i]
        print(pt)
        pred = neuralnetwork(pt[0],pt[1],w1,w2,b)
        if pred>0.5:
           print('Happy : ',pred)
           predicted_results.append(1)
        else:
           print('Sad : ',pred)
           predicted_results.append(0)
    #refer pic-8 in output.doc

    print(count)
    accuracy = (len(dt)-count)/(len(dt))
    print(accuracy)
    #refer pic-9 in output.doc
    
    #connect to visual cloud ,refer pic-10,11,12,13,14,15 in output.doc
    
    #Change of accuracy to 60%
    #print(new_arr)
    df['result']=new_arr
    ldata = len(df)
    print(ldata)
    #60% training 40% test ------- fixing length of training and test dataset {not needed}
    trainlen = int(ldata*0.6)
    testlen = ldata-trainlen
    i=0
    
    #we are going to work with energy , valence , mood ---> x , y, output
    
    energy = np.array(df['energy'])
    valence = np.array(df['valence'])
    output = np.array(df['result'])
    print(energy,valence,output)
    #refer pic-16 in output.doc
     
     #accuracy
     
     print(count)
     accuracy = (len(dt)-count)/(len(dt))
     print(accuracy)
     #refer pic-17 in output.doc
     
     #Change of accuracy to 40%
    #print(new_arr)
    df['result']=new_arr
    ldata = len(df)
    print(ldata)
    #40% training 60% test ------- fixing length of training and test dataset {not needed}
    trainlen = int(ldata*0.4)
    testlen = ldata-trainlen
    i=0
    
    #we are going to work with energy , valence , mood ---> x , y, output
    
    energy = np.array(df['energy'])
    valence = np.array(df['valence'])
    output = np.array(df['result'])
    print(energy,valence,output)
    #refer pic-18 in output.doc
     #accuracy

     
    print(count)
    accuracy = (len(dt)-count)/(len(dt))
    print(accuracy)
    #refer pic-19 in output.doc
     
     