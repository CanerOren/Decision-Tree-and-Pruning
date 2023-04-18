from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import category_encoders as ce
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

#DOSYA OKUNMASI VE AYIRMA İŞLEMLERİ
x_train=pd.read_csv("trainSet.csv")
y_train=x_train['class']
x_train=x_train.drop('class',axis=1)

x_test=pd.read_csv("testSet.csv")
y_test=x_test['class']
x_test=x_test.drop('class',axis=1)

#KATEGORİK VERİLER İÇİN ENCODER KULLANIMI
encoder=ce.OrdinalEncoder(cols=['A1','A4','A5','A6','A7','A9','A10','A12','A13'])

x_train=encoder.fit_transform(x_train)
x_test=encoder.fit_transform(x_test)

#MODELİN EĞİTİLMESİ
dt=DecisionTreeClassifier(criterion='gini',random_state=0)
train=dt.fit(x_train,y_train)

#Test Accuracy
y_pred_test=dt.predict(x_test)
acc_test=accuracy_score(y_test, y_pred_test)

#Train Accuracy
y_pred_train=dt.predict(x_train)
acc_train=accuracy_score(y_train,y_pred_train)

#TRAIN VE TEST SONUCU ELDE EDİLEN SKORLAR
cm_test=confusion_matrix(y_test,y_pred_test)
test_tn=cm_test[0][0]
test_tn_rate=test_tn/(test_tn+cm_test[0][1])
test_tp=cm_test[1][1]
test_tp_rate=test_tp/(test_tp+cm_test[1][0])


cm_train=confusion_matrix(y_train,y_pred_train)
train_tn=cm_train[0][0]
train_tn_rate=train_tn/(train_tn+cm_train[0][1])
train_tp=cm_train[1][1]
train_tp_rate=train_tp/(train_tn+cm_train[1][0])

#GÖRSELLEŞTİRİLMESİ
rep_text=tree.export_text(train)
print(rep_text)
plt.figure(figsize=(20,20))
features=x_train.columns
classes=["bad","good"]
tree.plot_tree(train,feature_names=features,class_names=classes,filled=True)
plt.show()

print("Eğitim (Train) Sonucu:")
print("Accuracy:",acc_train)
print("TPrate:",train_tp_rate)
print("TNrate:",train_tn_rate)
print("TP adedi:",train_tp)
print("TN adedi:",train_tn)
print("*******************")
print("Sınama (Test) Sonucu:")
print("Accuracy:",acc_test)
print("TPrate:",test_tp_rate)
print("TNrate:",test_tn_rate)
print("TP adedi:",test_tp)
print("TN adedi:",test_tn)

with open('sonuc.txt','w') as f:
    f.write("Eğitim (Train) Sonucu:\n")
    f.write("Accuracy: ")
    f.write(str(acc_train))
    f.write("\nTPrate: ")
    f.write(str(train_tp_rate))
    f.write("\nTNrate: ")
    f.write(str(train_tn_rate))
    f.write("\nTP adedi: ")
    f.write(str(train_tp))
    f.write("\nTN adedi: ")
    f.write(str(train_tn))
    f.write("\n*******************\n")
    f.write("\nSınama (Test) Sonucu:\n")
    f.write("Accuracy: ")
    f.write(str(acc_test))
    f.write("\nTPrate: ")
    f.write(str(test_tp_rate))
    f.write("\nTNrate: ")
    f.write(str(test_tn_rate))
    f.write("\nTP adedi: ")
    f.write(str(test_tp))
    f.write("\nTN adedi: ")
    f.write(str(test_tn))
    f.close

with open('graph.txt','w') as f:
    f.write(rep_text)
    f.close

#POST PRUNİNG İLE
path=dt.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities= path.ccp_alphas, path.impurities

dts=[]
for ccp_alpha in ccp_alphas:
    dt=tree.DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    dt.fit(x_train,y_train)
    dts.append(dt)

dts=dts[:-1]
ccp_alphas=ccp_alphas[:-1]
node_counts=[dt.tree_.node_count for dt in dts]
depth=[dt.tree_.max_depth for dt in dts]

plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='yaprak sayısı',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='derinlik',drawstyle="steps-post")
plt.xlabel("Alpha Değeri")
plt.ylabel("Yaprak Sayısı")
plt.legend()
plt.show()
#GRAFİKTETE GÖZÜKTÜĞÜ GİBİ YAPRAK SAYISI DÜŞTÜKÇE ALFA DEĞERİ ARTIYOR

pp_train_acc=[]
pp_test_acc=[]

for i in dts:
    pp_y_train_pred=i.predict(x_train)
    pp_y_test_pred=i.predict(x_test)
    pp_train_acc.append(accuracy_score(pp_y_train_pred,y_train))
    pp_test_acc.append(accuracy_score(pp_y_test_pred,y_test))
plt.scatter(ccp_alphas,pp_train_acc)
plt.scatter(ccp_alphas,pp_test_acc)
plt.plot(ccp_alphas,pp_train_acc,label='eğitim accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,pp_test_acc,label='test accuracy',drawstyle="steps-post")
plt.xlabel("Alpha değeri")
plt.ylabel("Accuracy")
plt.xticks(np.arange(0,0.020,0.002))
plt.legend()
plt.show()

#alpha=0.008 değeri olarak seçelim

dt_=tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.008)

pp_train=dt_.fit(x_train,y_train)
y_train_pred=dt_.predict(x_train)
y_test_pred=dt_.predict(x_test)

pp_acc_train=accuracy_score(y_train_pred,y_train)
pp_acc_test=accuracy_score(y_test_pred,y_test)

pp_cm_train=confusion_matrix(y_train_pred,y_train)
pp_train_tn=pp_cm_train[0][0]
pp_train_tn_rate=pp_train_tn/(pp_train_tn+pp_cm_train[0][1])
pp_train_tp=pp_cm_train[1][1]
pp_train_tp_rate=pp_train_tp/(pp_train_tn+pp_cm_train[1][0])

pp_cm_test=confusion_matrix(y_test_pred,y_test)
pp_test_tn=pp_cm_test[0][0]
pp_test_tn_rate=pp_test_tn/(pp_test_tn+pp_cm_test[0][1])
pp_test_tp=pp_cm_test[1][1]
pp_test_tp_rate=pp_test_tp/(pp_test_tp+pp_cm_test[1][0])


print("\n\n****** PRUNİNG SONRASI********\n\n")
pp_rep_text=tree.export_text(pp_train)
print(pp_rep_text)
plt.figure(figsize=(20,20))
features=x_train.columns
classes=["bad","good"]
tree.plot_tree(pp_train,feature_names=features,class_names=classes,filled=True)
plt.show()

print("Eğitim (Train) Sonucu:")
print("Accuracy:",pp_acc_train)
print("TPrate:",pp_train_tp_rate)
print("TNrate:",pp_train_tn_rate)
print("TP adedi:",pp_train_tp)
print("TN adedi:",pp_train_tn)
print("*******************")
print("Sınama (Test) Sonucu:")
print("Accuracy:",pp_acc_test)
print("TPrate:",pp_test_tp_rate)
print("TNrate:",pp_test_tn_rate)
print("TP adedi:",pp_test_tp)
print("TN adedi:",pp_test_tn)


with open('pruning_sonrasi_sonuc.txt','w') as f:
    f.write("Eğitim (Train) Sonucu:\n")
    f.write("Accuracy: ")
    f.write(str(pp_acc_train))
    f.write("\nTPrate: ")
    f.write(str(pp_train_tp_rate))
    f.write("\nTNrate: ")
    f.write(str(pp_train_tn_rate))
    f.write("\nTP adedi: ")
    f.write(str(pp_train_tp))
    f.write("\nTN adedi: ")
    f.write(str(pp_train_tn))
    f.write("\n*******************\n")
    f.write("\nSınama (Test) Sonucu:\n")
    f.write("Accuracy: ")
    f.write(str(pp_acc_test))
    f.write("\nTPrate: ")
    f.write(str(pp_test_tp_rate))
    f.write("\nTNrate: ")
    f.write(str(pp_test_tn_rate))
    f.write("\nTP adedi: ")
    f.write(str(pp_test_tp))
    f.write("\nTN adedi: ")
    f.write(str(pp_test_tn))
    f.close
with open('pruning_sonrasi_graph.txt','w') as f:
    f.write(pp_rep_text)
    f.close