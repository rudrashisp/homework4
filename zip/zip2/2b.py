import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
np.random.seed(1234)
X=np.random.randint(low=-40,high=41,size=(200,2))
Y= np.empty(200)
X = np.concatenate([X, np.ones(len(X))[:, np.newaxis]], axis=1)
for i in range(200):
    if((X[i][0]+3*X[i][1]-2)>0):
        Y[i]=1.0
    else:
        Y[i]=-1.0
W=np.random.random_sample(3)
delW=np.zeros(3)
acc_list,error_list=[],[]
eta=0.001
thresh=0.5
for k in range(50):
    o=np.matmul(X,W)
    for i in range(200):
        delW=delW+eta*(Y[i]-o[i])*X[i,:]
    W=np.add(W,delW)
    predict=np.matmul(X,W)
    for j in range(200):
        if(predict[j]>0):
            predict[j]=1.0
        else:
            predict[j]=-1.0
    acc = sum([a==b for a,b in zip(Y,predict)])/len(X)
    acc_list += [acc]
    error_list +=[1-acc]
    if(k>0):
        if(error_list[k]-error_list[k-1]>thresh):
            W-=delW
            eta*=0.95
        if(error_list[k]<error_list[k-1]):
            eta*=1.134
print(Y)
print(predict)
t=[i+1 for i in range(50)]

plt.xlabel("number of epochs")
plt.ylabel("Train Error")
plt.title('Adaptive Learning rate')
plt.plot(t,error_list)
plt.show()