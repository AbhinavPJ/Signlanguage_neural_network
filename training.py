import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self,input_features=10,h1=50,h2=50,output=28):
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
df=pd.read_csv('result_cleaned.csv')
mapping = {
    'A': 0.0, 'B': 1.0, 'C': 2.0, 'D': 3.0, 'E': 4.0, 'F': 5.0, 'G': 6.0,
    'H': 7.0, 'I': 8.0, 'J': 9.0, 'K': 10.0, 'L': 11.0, 'M': 12.0, 'N': 13.0,
    'O': 14.0, 'P': 15.0, 'Q': 16.0, 'R': 17.0, 'S': 18.0, 'T': 19.0,
    'U': 20.0, 'V': 21.0, 'W': 22.0, 'X': 23.0, 'Y': 24.0, 'Z': 25.0,
    'del': 26.0, 'space': 27.0
}

df['label'] = df['label'].map(mapping)
x=df.drop('label',axis=1)
y=df['label']
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
