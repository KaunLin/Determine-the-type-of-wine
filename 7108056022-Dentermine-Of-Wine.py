
# coding: utf-8

# In[1]:


import numpy as np
from numpy import *
from numpy.linalg import  *


# # 讀取資料轉成numpy裡的array(資料前處理)

# ## 在創建另一個資料檔，沒有第一排類別，用來算變異數

# In[2]:


data = []
with open("C:/Users/qscf6/Desktop/wine.data", "r") as f:
    lines = f.readlines()
    for line in lines:
        a = line.split(",")
        a[-1] = a[-1].split("\n")[0]
        a = list(map(float , a))
        data.append(np.array(a))
covdata = []
with open("C:/Users/qscf6/Desktop/winedata.data", "r") as f:
    lines = f.readlines()
    for line in lines:
        a = line.split(",")
        a[-1] = a[-1].split("\n")[0]
        a = list(map(float , a))
        covdata.append(np.array(a))


# # 統計每個類別有幾筆資料並算出先驗機率

# In[3]:


length = len(data)
count1 = 0
count2 = 0
count3 = 0
for i in range (length):
    if data[i][0] == 1:
        count1 = count1 + 1
    elif data[i][0] == 2:
        count2 = count2 + 1
    else:
        count3 = count3 + 1
print(count1)
P1 = count1 / length
print(P1)
print(count2)
P2 = count2 / length
print(P2)
print(count3)
P3 = count3 / length
print(P3)


# # 將每個類別資料取一半當作訓練資料

# # 算出每個類別的13個特徵的平均數與變異數

# ## 第一類

# In[4]:


data1 = []
averge1 = []
sd1 = []
for i in data[:int(count1/2)]:
    data1.append(i)
for j in range(1,14):
    feature = []
    for i in range (0,len(data1)):
        feature.append(data1[i][j])
    averge1.append(np.mean(feature))
averge1 = np.array(averge1)
print(averge1)
covdata1 = []
for i in covdata[:int(count1/2)]:
    covdata1.append(i)
covdata1 = np.array(covdata1)
covdata1 = covdata1.T
sd1 = np.cov(covdata1)
inversesd1 = np.linalg.inv(sd1)
print(sd1)


# ## 第二類

# In[5]:


data2 = []
averge2 = []
sd2 = []
for i in data[count1:(count1 + int(count2/2))]:
    data2.append(i)
for j in range(1,14):
    feature = []
    for i in range (len(data2)):
        feature.append(data2[i][j])
    averge2.append(np.mean(feature))
averge2 = np.array(averge2)
print(averge2)
covdata2 = []
for i in covdata[count1:(count1 + int(count2/2))]:
    covdata2.append(i)
covdata2 = np.array(covdata2)
covdata2 = covdata2.T
sd2 = np.cov(covdata2)
inversesd2 = np.linalg.inv(sd2)
print(sd2)


# ## 第三類

# In[6]:


data3 = []
averge3 = []
sd3 = []
for i in data[count1+count2:(count1+count2 + int(count3/2))]:
    data3.append(i)
for j in range(1,14):
    feature = []
    for i in range (len(data3)):
        feature.append(data3[i][j])
    averge3.append(np.mean(feature))
    sd3.append(np.cov(feature))
averge3 = np.array(averge3)
print(averge3)
covdata3 = []
for i in covdata[count1+count2:(count1+count2 + int(count3/2))]:
    covdata3.append(i)
covdata3 = np.array(covdata3)
covdata3 = covdata3.T
sd3 = np.cov(covdata3)
inversesd3 = np.linalg.inv(sd3)
print(sd3)


# # 由上面每類的平均值和變異數，求出隨機某一筆測試資料x屬於哪一類

# In[12]:


x = []
x = np.array([13.56,1.73,2.46,20.5,116,2.96,2.78,.2,2.45,6.25,.98,3.03,1120])
one = 0
one = -(np.log(P1)) + 0.5*(((x-averge1).T).dot(inversesd1).dot(x-averge1))+ 0.5*(det(sd1))
two = 0
two =  -(np.log(P2)) + 0.5*(((x-averge2).T).dot(inversesd2).dot(x-averge2))+ 0.5*(det(sd2))
three = 0
three = -(np.log(P3)) + 0.5*(((x-averge3).T).dot(inversesd3).dot(x-averge3))+ 0.5*(det(sd3))
answer = min(one,two,three)
if answer == one:
    print("one")
elif answer == two:
    print("two")
else:
    print("three")


# # 使用測試資料，算出準確度

# In[9]:


correct = 0
mistake = 0
test = []
for i in data[int(count1/2):count1]:
    test.append(i)
for i in data[count1+int(count2/2):count1+count2]:
    test.append(i)
for i in data[count1+count2+int(count3/2):count1+count2+count3]:
    test.append(i)
for i in test:
    xi = []
    for j in range(1,14):
        xi.append(np.array(i[j]))
    first = -(np.log(P1)) + 0.5*(((xi-averge1).T).dot(inversesd1).dot(xi-averge1))+ 0.5*(det(sd1))
    second = -(np.log(P2)) + 0.5*(((xi-averge2).T).dot(inversesd2).dot(xi-averge2))+ 0.5*(det(sd2))
    third = -(np.log(P3)) + 0.5*(((xi-averge3).T).dot(inversesd3).dot(xi-averge3))+ 0.5*(det(sd3))
    an = min(first,second,third)
    if an == first:
        an = 1.0
    elif an == second:
        an = 2.0
    else:
        an = 3.0
    if an == i[0]:
        correct = correct + 1
    else:
        mistake = mistake + 1
print(correct)
rate = (float(correct) / float(len(test)))
print(rate)

