import os
import pynlpir
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
vec = DictVectorizer()
vct=[]
pynlpir.open()

def readF(path, n=0):
    for file in os.listdir(path):
        f=open(path+file,'r')
        s=f.read()
        x=pynlpir.get_key_words(s,weighted=True)
        dic={}
        for i in x:
            dic[i[0]]=i[1]
        vct.append(dic)

def makeVector(pathP,pathN,pathT):
    readF(pathP)
    readF(pathN)
    return [vec.fit_transform(vct).toarray(),len(os.listdir(pathP)),len(os.listdir(pathN)),len(os.listdir(pathT))]

readF('./Target/')
Target=vec.fit_transform(vct).toarray()
temp=makeVector('./Pos/','./Neg/','./Target/')
len1=temp[1]
len2=temp[2]
len3=temp[3]
Eg = temp[0][len3:]
Cls = [0]*len2+[1]*len1    # 0 for non-edu and 1 for edu
classify = svm.SVC()
classify.fit(Eg, Cls)
targ=vec.fit_transform(vct).toarray()
print(targ.shape,temp[0].shape,sep='\n\n')
print(classify.predict(targ[0-len3:]))
pynlpir.close()
