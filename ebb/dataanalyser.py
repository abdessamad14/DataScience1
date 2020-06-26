# Author: Abdessamad Abba, Goethe-Universitaet Frankfurt am Main, FB 12 - Informatik und Mathematik
# Mail: s4295146@stud.uni-frankfurt.de
# Matrikelnummer: 5776119
# Purpose: Abruf von enddata_labeled.csv als Input.Er trainiert un test das MLP und liefert qualitativen Regeln sowie Modellen aus. 
# Agent: Dataanalyser
# Date: 15.05.2019

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn import metrics
from sklearn import preprocessing, model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from numpy import array
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from decimal import Decimal
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from os import system, name 
from time import sleep
from pyfiglet import Figlet
import bisect
import sys
import networkx as nx
from tensorflow import keras
from tensorflow.keras import layers

sys.setrecursionlimit(10000)
matplotlib.style.use('ggplot')

########################--Begin--#############################
########################Classification########################




data = pd.read_csv('./Data/enddata_labeled.csv')


X = data[["'Bewoelkung'", "'Temperatur'", "'Windgeschwindigkeit'", "'Oberflaechendruck'","'o3'", "'no2'"]].iloc[4:]
y = data["'O3Klasse'"].values[:-4]

'''
Datensatz staerken, indem Abweichungen zu jeweiligen Inputs addiert wird
paramater: to_make_strong_list, big_list, training_input_list, klasse
return: temp_list2
'''
def make_training_strong(to_make_strong_list,big_list,training_input_list,klasse):
	
	temp_list1=[]
	temp_list2=[]
	count = 0
	#0.00001
	for i in to_make_strong_list[0]:
		while count < round((len(big_list[0])-len(to_make_strong_list[0]))/len(to_make_strong_list[0])):
			if count <= round((len(big_list[0])-len(to_make_strong_list[0]))/(2*len(to_make_strong_list[0]))):
				training_input_list[i][0]=training_input_list[i][0]+0.001
				temp_list1.append(training_input_list[i][0])
				training_input_list[i][1]=training_input_list[i][1]+0.05
				temp_list1.append(training_input_list[i][1])
				training_input_list[i][2]=training_input_list[i][2]+0.005
				temp_list1.append(training_input_list[i][2])
				training_input_list[i][3]=training_input_list[i][3]+0.2
				temp_list1.append(training_input_list[i][3])
				training_input_list[i][4]=training_input_list[i][4]+0.05
				temp_list1.append(training_input_list[i][4])
				training_input_list[i][5]=training_input_list[i][5]+0.05
				temp_list1.append(training_input_list[i][5])
				temp_list1.append(klasse)
				temp_list2.append(temp_list1)
				temp_list1=[]
				count = count + 1
			else:
				training_input_list[i][0]=training_input_list[i][0]-0.001
				temp_list1.append(training_input_list[i][0])
				training_input_list[i][1]=training_input_list[i][1]-0.05
				temp_list1.append(training_input_list[i][1])
				training_input_list[i][2]=training_input_list[i][2]-0.005
				temp_list1.append(training_input_list[i][2])
				training_input_list[i][3]=training_input_list[i][3]-0.2
				temp_list1.append(training_input_list[i][3])
				training_input_list[i][4]=training_input_list[i][4]-0.05
				temp_list1.append(training_input_list[i][4])
				training_input_list[i][5]=training_input_list[i][5]-0.05
				temp_list1.append(training_input_list[i][5])
				
				temp_list1.append(klasse)
				temp_list2.append(temp_list1)
				temp_list1=[]
				count = count + 1
				
		count=0
	return temp_list2


'''
Funktion, die den korregierten Datensatz in Form von Listen liefert.
paramater: input_data, output_data
return: strong_total_list, strong_total_list_output
'''
def getStrongData(input_data,output_data):
	list_Zielwert_Gelb=np.where(output_data == 'Zielwert_Gelb')
	list_Zielwert_Rot=np.where(output_data == 'Zielwert_Rot')

	list_Informationsschwelle_Gruen=np.where(output_data == 'Informationsschwelle_Gruen')
	list_Informationsschwelle_Gelb=np.where(output_data == 'Informationsschwelle_Gelb')
	list_Informationsschwelle_Rot=np.where(output_data == 'Informationsschwelle_Rot')

	list_Alarmschwelle_Stufe1=np.where(output_data =='Alarmschwelle_Stufe1')
	list_Alarmschwelle_Stufe2=np.where(output_data =='Alarmschwelle_Stufe2')

	list2=index_list_ir=np.where(output_data == 'Zielwert_Gruen')
	list3=input_data.values.tolist()



	add_list_to_train_zge=make_training_strong(list_Zielwert_Gelb,list2,list3,'Zielwert_Gelb')
	add_list_to_train_zgr=make_training_strong(list_Zielwert_Rot,list2,list3,'Zielwert_Rot')

	add_list_to_train_igr=make_training_strong(list_Informationsschwelle_Gruen,list2,list3,'Informationsschwelle_Gruen')
	add_list_to_train_ige=make_training_strong(list_Informationsschwelle_Gelb,list2,list3,'Informationsschwelle_Gelb')
	add_list_to_train_iro=make_training_strong(list_Informationsschwelle_Rot,list2,list3,'Informationsschwelle_Rot')

	add_list_to_train_a1=make_training_strong(list_Alarmschwelle_Stufe1,list2,list3,'Alarmschwelle_Stufe1')
	add_list_to_train_a2=make_training_strong(list_Alarmschwelle_Stufe2,list2,list3,'Alarmschwelle_Stufe2')

	strong_total_list=[]
	strong_total_list_output=[]
	for i in list3:
		strong_total_list.append(i)

	for i in add_list_to_train_zge:
		strong_total_list.append(i[:-1])

	for i in add_list_to_train_zgr:
		strong_total_list.append(i[:-1])

	for i in add_list_to_train_igr:
		strong_total_list.append(i[:-1])

	for i in add_list_to_train_ige:
		strong_total_list.append(i[:-1])

	for i in add_list_to_train_iro:
		strong_total_list.append(i[:-1])

	for i in add_list_to_train_a1:
		strong_total_list.append(i[:-1])

	for i in add_list_to_train_a2:
		strong_total_list.append(i[:-1])

	####Output

	for i in y:
		strong_total_list_output.append(i)
	for i in add_list_to_train_zge:
		strong_total_list_output.append(i[-1])

	for i in add_list_to_train_zgr:
		strong_total_list_output.append(i[-1])

	for i in add_list_to_train_igr:
		strong_total_list_output.append(i[-1])

	for i in add_list_to_train_ige:
		strong_total_list_output.append(i[-1])

	for i in add_list_to_train_iro:
		strong_total_list_output.append(i[-1])

	for i in add_list_to_train_a1:
		strong_total_list_output.append(i[-1])

	for i in add_list_to_train_a2:
		strong_total_list_output.append(i[-1])

	return strong_total_list, strong_total_list_output
	

######Final Lists
X, y= getStrongData(X, y)
print(X)
strong_total_list=X
strong_total_list_output=y
######Final Lists

'''
Output in Form von Vektoren kodieren;
Input zwischen 0 und 1 skalieren;
Input und Output aufteilen: 80% Train und 20% Test
parameter:x, j
return: tupple (train_x, test_x, train_y, test_y, scaler,encoder)
'''
def preparData(x,j):
	x = np.array([np.array(a) for a in x])
	encoder = LabelEncoder()
	encoder.fit(j)
	j = encoder.transform(j)
	j = np_utils.to_categorical(j)

	scaler=StandardScaler()
	x=scaler.fit_transform(x)

	train_x, test_x, train_y, test_y = model_selection.train_test_split(x,j,test_size = 0.2, random_state = 0)
	tupple = (train_x, test_x, train_y, test_y, scaler,encoder)
	return tupple

'''
Data mit neuem Model trainieren und die Vorhersage liefern
parameter: x,xt,j,jt,enc
retrun: prediction_, model
'''
def trainandtest(x,xt,j,jt,enc):	
	#model = keras.Sequential()
	#model.add(Dense(8, input_dim = 5 , activation = 'relu'))
	#model.add(Dense(16, activation = 'relu'))
	#model.add(Dense(32, activation = 'relu'))
	#model.add(Dense(64, activation = 'relu'))
	#model.add(Dense(8, activation = 'softmax'))
	
	model = keras.Sequential(
    	[
        layers.Dense(6, activation="relu", name="layer1"),
        layers.Dense(8, activation="relu", name="layer2"),
        layers.Dense(16, activation="relu", name="layer3"),
        layers.Dense(32, activation="relu", name="layer4"),
        layers.Dense(64, activation="relu", name="layer5"),
        layers.Dense(8, activation="softmax", name="layer3"),
   	]
	)

	model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

	model.fit(x, j, epochs = 15, batch_size = 2)

	scores = model.evaluate(xt, jt)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	predictions = model.predict_classes(xt)
	prediction_ = np.argmax(to_categorical(predictions), axis = 1)
	prediction_ = enc.inverse_transform(prediction_)
	print("TRAINING FINISCHED")
	#ask_for_saving(model)
	return prediction_, model
	
'''
Data mit vorhandenem Model trainieren und die Vorhersage liefern
parameter: x,j,enc
return: prediction_
'''
def ask_for_saving(mod):
	frage1 = raw_input("Do you want to save our trained model? Yes[Y] No[N]: ")
	if frage1 == 'Y':
		frage2 = raw_input("ATTENTION! YOU WILL LOSE ALL TRAINED MODELS, Continue? Yes[Y] No[N]: ")
		if frage2 == 'Y':
			savemodel(mod)
		elif frage2 == 'N':
			pass
		else:
			print("PLEASE GIVE A VALID INPUT!")
			ask_for_saving(mod)
	elif frage1 == 'N':
		pass
	else:
		print("PLEASE GIVE A VALID INPUT!")
		ask_for_saving(mod)

def loadandtest(x,j,enc):
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

	# evaluate loaded model on test data
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	score = loaded_model.evaluate(x, j)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
	predictions = loaded_model.predict_classes(x)
	prediction_ = np.argmax(to_categorical(predictions), axis = 1)
	prediction_ = enc.inverse_transform(prediction_)
	print("LOADING FINISCHED")
	return prediction_

'''
Trained Model mit weight als JSON File im Projektverzeichnis speichern
parameter: mod
'''
def savemodel(mod):
	#serialize model to JSON
	model_json = mod.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	mod.save_weights("model.h5")
	print("Saved model to disk")

'''
Skalierte Input Data zu dem ursprunglichen Input invertieren
parameter: x, scaler
return: x
'''
def inversencoding_text(x,scaler):
	x =scaler.inverse_transform(x)
	return x
	
########################--End--###############################
########################Classification########################

var1 = preparData(X,y)[0] 
var2 = preparData(X,y)[1] 
var3 = preparData(X,y)[2]
var4 = preparData(X,y)[3]
var5 = preparData(X,y)[4]
var6 = preparData(X,y)[5]

predictions, trainedmodel = trainandtest(var1,var2,var3,var4,var6)	


########################--Begin--#########################
########################Regression########################

#####Strong List to Dataframe
list_to_dataframe=[]
tempolist=[]
for i in range(len(strong_total_list)):
	tempolist=strong_total_list[i]
	tempolist.append(strong_total_list_output[i])
	list_to_dataframe.append(tempolist)


df4= pd.DataFrame(list_to_dataframe,  columns=["'Bewoelkung'", "'Temperatur'", "'Windgeschwindigkeit'", "'Oberflaechendruck'","'o3'", "'no2'","'O3_Klasse'"])
df4.to_csv("./Data/enddata_strong.csv",  encoding='utf-8')
#####Strong List to Dataframe


data = pd.read_csv('./Data/enddata_strong.csv')

X = data[["'Bewoelkung'", "'Temperatur'", "'Windgeschwindigkeit'", "'Oberflaechendruck'", "'o3'", "'no2'"]].iloc[4:]
y = data["'o3'"].values[:-4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  

y_pred = regressor.predict(X_test)


'''
Regressionergebnisse einzeigen
parameter: test, pred, wieviel
'''
def show_regression_result(test,pred,wieviel):
	df = pd.DataFrame({'Actual': test, 'Predicted': pred})
	print(df)
	df1 = df.head(wieviel)
	print('Mean Absolute Error:', metrics.mean_absolute_error(test, pred))  
	print('Mean Squared Error:', metrics.mean_squared_error(test, pred))  
	print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test, pred)))

dataset = pd.read_csv('./Data/enddata_labeled.csv')


show_prediction_bar(y_test,y_pred,20)
########################--End--###########################
########################Regression########################
