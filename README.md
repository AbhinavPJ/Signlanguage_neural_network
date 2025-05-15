Project objective: Use neural network to convert realtime image which represents a sign-language symbol(letters only,no words) to the letter

Overview:
I took a dataset of 80k images from kaggle each named with the equivalent alphabet as filename.Using cv2, I analyzed the position of the fingertips of each finger relative to the base of the wrist. I converted this position vector to polar coordinates and thus got 10 input features for my neural network.I converted the 80k images to a csv file by randomly selecting 20k images(16K train,4k test) and storing their position vector info in a csv file along with their correct answers.To analyze this, I used a linear CNN(with pytorch ofc) with two hidden layers each having 50 neurons,then a layer of output features having 28 neurons(26 alphabets,del,space). I trained this model using the dataset,using a similar code as that used in the iris flower classifier(classify flowers into 3 species using sepal width,petal width etc.)
I also plotted the entropy loss during this and observed a decreasing curve,this confirmed that the neural network was getting properly trained.
I then verified the correctness by looking at the next 100 images after the training set to see how many were correctly guessed,and I saw most of them were indeed correct.Then,I ran the evaluation code and got the below result.
Finally I used this neural network to see the letters as I was signalling in real time.It worked!!!

Evaluation parameters:
Accuracy: 94.15%
precision recall average instances
A 0.95 0.98 0.97 101
B 0.97 1.00 0.98 86
C 0.99 0.98 0.98 100
D 0.99 0.95 0.97 134
E 0.98 0.93 0.95 122
F 0.96 0.99 0.98 129
G 0.99 1.00 1.00 136
H 0.95 0.97 0.96 107
I 0.93 0.95 0.94 116
J 0.96 0.94 0.95 113
K 0.97 0.97 0.97 117
L 0.96 1.00 0.98 105
M 0.74 0.80 0.77 66
N 0.88 0.71 0.79 63
O 0.92 1.00 0.96 106
P 0.86 0.87 0.87 102
Q 0.88 0.87 0.87 99
R 0.79 0.93 0.85 118
S 0.97 0.98 0.97 128
T 1.00 0.94 0.97 100
U 0.79 0.68 0.73 87
V 0.98 0.91 0.95 126
W 0.99 0.96 0.97 115
X 0.95 0.98 0.97 85
Y 0.99 0.99 0.99 128
Z 0.98 0.96 0.97 109
del 0.91 0.93 0.92 73
space 0.97 0.97 0.97 68

    accuracy                           0.94      2939
