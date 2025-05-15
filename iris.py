import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self,input_features=4,h1=8,h2=9,output=3):
        super().__init__()
        self.fc1=nn.Linear(input_features,h1)
        self.fc2=nn.Linear(h1,h2)
        self.out=nn.Linear(h2,output)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.out(x)
        return x
torch.manual_seed(32)
model=Model()
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
df['species']=df['species'].map({'setosa':0.0,'versicolor':1.0,'virginica':2.0})
x=df.drop('species',axis=1)
y=df['species']
x=x.values
y=y.values
losses=[]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)
x_train=torch.FloatTensor(x_train)
x_test=torch.FloatTensor(x_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
print(model.parameters())
for i in range(100):
  y_pred=model.forward(x_train)
  loss=criterion(y_pred,y_train)
  print(loss)
  losses.append(loss.detach().numpy())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
plt.plot(range(100),losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
print(model.parameters())
with torch.no_grad():
  for i,data in enumerate(x_test):
    y_val=model.forward(data)
    if y_val.argmax().item()==y_test[i]:
      print('Correct')
    else:
      print('Wrong')
    #print(i+1,y_val,y_test[i])
print(loss)