# -*- coding: utf-8 -*-
# Author: Abdessamad Abba, Goethe-Universitaet Frankfurt am Main, FB 12 - Informatik und Mathematik
# Mail: s4295146@stud.uni-frankfurt.de
# Matrikelnummer: 5776119
# Purpose: Abruf von my_locations_meteo.csv, O3.csv und NO2.csv als Input. Die Vollstaendigkeit der Daten pruefen, dann die meteorologischen sowie chemischen Daten zusammenfuegen 
# Agent: Dataparser
# Date: 15.05.2019

import pandas as pd
import numpy as np
import math
import re
import sys
from os import system, name 
from time import sleep 
from pyfiglet import Figlet
sys.setrecursionlimit(10000)

#Meteorologische Daten auf /Data Verzeichnis aufrufen
df_meteo= pd.read_csv("./Data/meteorological_data.csv",header=None, usecols=[0])
df_meteo.columns = [''] * len(df_meteo.columns)
df_meteo = df_meteo[2:23354]
new = df_meteo[''].str.split("    ", n = 8, expand = True)
df_meteo["Date"]= new[0]
df_meteo["Time"]= new[1]
df_meteo["Parameter Name"]= new[3]
df_meteo["Value at MetPoint"]= new[5]
df_meteo.drop(columns =[''], inplace = True) 
df_meteo.dropna(inplace = True) 

#Struktur von meteorological_data.csv aendern
df_meteo['Date']= pd.to_numeric(df_meteo['Date'])
df_meteo['Time']= pd.to_numeric(df_meteo['Time'])
df_meteo['Value at MetPoint']= pd.to_numeric(df_meteo['Value at MetPoint'])

#Meteorologische Daten absteigend sortieren gemaess Datum und Zeit
df_meteo_sorted = df_meteo.sort_values(by=["Date","Time"], ascending=False)


'''
Header von CSV-Output erstellen.
return: header_list
'''
def constract_header():
	header_list=["'Datum'","'Bewoelkung'","'Temperatur'","'Windgeschwindigkeit'","'Oberflaechendruck'","'o3'","'no2'","'O3Klasse'"]
	return header_list

'''
Temperature Spalte zu einer Liste konvertieren
parameter: df > CSV von meterologischen Daten
return: temp_list 
'''
def get_list_temperature(df):
	list_values = ["2 metre temperature  2t", " 2 metre temperature  2t", "   2 metre temperature  2t"]
	temp_list = df.loc[df["Parameter Name"].isin(list_values)]["Value at MetPoint"].tolist()
	return temp_list

'''
Bewoelkung Spalte zu einer Liste konvertieren
parameter: df > CSV von meterologischen Daten
return: cloud_cover_list 
'''
def get_list_cloud_cover(df):
	list_values = ["Total cloud cover  tcc", " Total cloud cover  tcc", "   Total cloud cover  tcc"]
	cloud_cover_list = df.loc[df["Parameter Name"].isin(list_values)]["Value at MetPoint"].tolist()
	return cloud_cover_list

'''
Oberflaechendruck Spalte zu einer Liste konvertieren
parameter: df > CSV von meterologischen Daten
return: surface_pressure_list 
'''

def get_list_surface_pressure(df):
	list_values = ["Surface pressure  sp", " Surface pressure  sp", "   Surface pressure  sp"]
	surface_pressure_list = df.loc[df["Parameter Name"].isin(list_values)]["Value at MetPoint"].tolist()
	return surface_pressure_list

'''
V und U wind component Spalte zu einer Liste konvertieren,
dann die Windgeschwindigkeit rechnen.
parameter: df > CSV von meterologischen Daten
return: wind_list 
'''
def get_list_wind(df):
	list_values_v = ["10 metre V wind component  10v", " 10 metre V wind component  10v", "   10 metre V wind component  10v"]
	list_values_u = ["10 metre U wind component  10u", " 10 metre U wind component  10u", "   10 metre U wind component  10u"]
	list_v_c_w = df.loc[df["Parameter Name"].isin(list_values_v)]["Value at MetPoint"].tolist()
	list_u_c_w = df.loc[df["Parameter Name"].isin(list_values_u)]["Value at MetPoint"].tolist()

	wind_list=[]
	for i in range(0, len(list_u_c_w)):
		x = list_v_c_w[i]
		y = list_u_c_w[i]
		erg_wind=math.sqrt(math.pow(x,2) + math.pow(y,2))
		wind_list.append(erg_wind)
	return wind_list


'''
Ozon Spalte zu einer Liste konvertieren
parameter: df > CSV von chemischen Daten
return: o3_list 
'''
def get_list_chem(df):
	list_zeit=[]
	no2_list = []
	o3_list = []
	zeit_list= df["Datum"].tolist()
	messwert_o3= df["o3"].tolist()
	messwert_no2= df["no2"].tolist()
	for i in zeit_list:
		if (("24:00" in i) or ("06:00" in i) or ("12:00" in i) or ("18:00" in i)):
			o3_list.append(messwert_o3[zeit_list.index(i)])
			no2_list.append(messwert_no2[zeit_list.index(i)])
	return list(reversed(o3_list)), list(reversed(no2_list))

#Chemische Daten auf /Data Verzeichnis aufrufen
df_chemic = pd.read_csv("./Data/chemische_Daten.csv",)


df_chemic = df_chemic[1:23353]
new = df_chemic['Stationscode;Datum;"Feinstaub (PM₁₀) stündlich gleitendes Tagesmittel in µg/m³";"Ozon (O₃) Ein-Stunden-Mittelwert in µg/m³";"Stickstoffdioxid (NO₂) Ein-Stunden-Mittelwert in µg/m³";Luftqualitätsindex'].str.split(";", n = 6, expand = True)
df_chemic["Datum"]= new[1]
df_chemic["o3"]= new[3]
df_chemic["no2"]= new[4]
df_chemic.drop(columns =['Stationscode;Datum;"Feinstaub (PM₁₀) stündlich gleitendes Tagesmittel in µg/m³";"Ozon (O₃) Ein-Stunden-Mittelwert in µg/m³";"Stickstoffdioxid (NO₂) Ein-Stunden-Mittelwert in µg/m³";Luftqualitätsindex'], inplace = True) 
df_chemic.dropna(inplace = True) 

'''
Datum Spalte zu Liste konvertieren dann,
Datum Format zu dieser Form aendern: yyyymmddhhmm(201812311800)
parameter: df > CSV von meteorologischen Daten
return: list_date_meteo
'''
def date_formated_meteo(df):
	list_date_meteo=[]
	for i in df["Date"].unique().tolist():
		list_date_meteo.append(str(i)+"1800")
		list_date_meteo.append(str(i)+"1200")
		list_date_meteo.append(str(i)+"0600")
		list_date_meteo.append(str(i)+"2400")
	list_date_meteo.sort(reverse=True)
	return list_date_meteo
	

'''
Datum Spalte zu Liste konvertieren dann,
Datum Format zu dieser Form aendern: yyyymmddhhmm(201812311800)
parameter: df > CSV von chemischen Daten
return: list_date_chemi
'''
def date_formated_chemi(df):
	list_date_chemi=[]
	for i in df["Datum"].tolist():
		if (("24:00" in i) or ("06:00" in i) or ("12:00" in i) or ("18:00" in i)):
			test=re.findall(r"[\w]+", i)
			i=test[2]+test[1]+test[0]+test[3]+test[4]
			list_date_chemi.append(i)
	return list(reversed(list_date_chemi))
	

'''
Liste von Tupeln in dieser Form erstellen: (Datum der Messung, der Wert o3, der Wert no2)
parameter: date_list(Liste von Datum der chemischen Daten), list_o3(Liste von Ozon Werte), list_no2(Liste von no2 Werte)
return: list_date_o3_tuple
'''
list_date_o3_tuple=[]
def get_tuple_o3(date_list_o3,list_o3,list_no2):
	if len(date_list_o3)==0:
	        return
	get_tuple_o3(date_list_o3[:-1], list_o3[:-1], list_no2[:-1])
	list_date_o3_tuple.append((date_list_o3[date_list_o3.index(date_list_o3[-1])], list_o3[date_list_o3.index(date_list_o3[-1])], list_no2[date_list_o3.index(date_list_o3[-1])]))
	return list_date_o3_tuple
	

'''
Liste von Tupeln in dieser Form erstellen: (Datum der Messung, der Wert Temperatur, der Wert Bewoelkung, der Wert Temperatur, der Wert Windgeschwindigkeit, der Wert Oberflaechendruck)
parameter: date_list(Liste von Datum der chemischen Daten), list_o3(Liste von Ozon Werte), list_no2(Liste von no2 Werte)
return: list_date_o3_tuple
'''
list_date_meteo_tuple=[]
def get_tuple_meteo(date_list_meteo,list_cc,list_temp,list_wind, list_sp):
	if len(date_list_meteo)==0:
	        return
	get_tuple_meteo(date_list_meteo[:-1], list_cc[:-1], list_temp[:-1], list_wind[:-1], list_sp[:-1])
	list_date_meteo_tuple.append((date_list_meteo[date_list_meteo.index(date_list_meteo[-1])], float(list_cc[date_list_meteo.index(date_list_meteo[-1])]), float(list_temp[date_list_meteo.index(date_list_meteo[-1])]), float(list_wind[date_list_meteo.index(date_list_meteo[-1])]), float(list_sp[date_list_meteo.index(date_list_meteo[-1])])))
	return list_date_meteo_tuple
	

'''
Format des Datums von meteorologischen Daten aendern
'''
ordred_list_meteo = []
for i in get_tuple_meteo(date_formated_meteo(df_meteo_sorted),get_list_cloud_cover(df_meteo_sorted),get_list_temperature(df_meteo_sorted),get_list_wind(df_meteo_sorted), get_list_surface_pressure(df_meteo_sorted)):
	i =list(i)
	if i[0][8:]== '0600':
		i[0] = i[0][:-4]+'2400'
	elif i[0][8:]== '1200':
		i[0] = i[0][:-4]+'0600'
	elif i[0][8:]== '1800':
		i[0] = i[0][:-4]+'1200'
	elif i[0][8:]== '2400':
		i[0] = i[0][:-4]+'1800'
	else:
		pass
	ordred_list_meteo.append(i)



ordred_list_meteo.sort(key=lambda x: x[0], reverse = True)

ordred_list_chemisch= get_tuple_o3(date_formated_chemi(df_chemic),get_list_chem(df_chemic)[0],get_list_chem(df_chemic)[1])


end_list=[]
'''
Data cleanup
'''
for i in range(len(ordred_list_meteo)):
	if ordred_list_chemisch[i][1] == '-' and ordred_list_chemisch[i][2]== '-':
		ordred_list_meteo[i].append(float(0))
		ordred_list_meteo[i].append(float(0))
	elif ordred_list_chemisch[i][1] == '-' and ordred_list_chemisch[i][2] != '-':
		ordred_list_meteo[i].append(float(0))
		ordred_list_meteo[i].append(float(ordred_list_chemisch[i][2]))
	elif ordred_list_chemisch[i][1] != '-' and ordred_list_chemisch[i][2] == '-':
		ordred_list_meteo[i].append(float(ordred_list_chemisch[i][1]))
		ordred_list_meteo[i].append(float(0))
	else:
		ordred_list_meteo[i].append(float(ordred_list_chemisch[i][1]))
		ordred_list_meteo[i].append(float(ordred_list_chemisch[i][2]))
	end_list.append(ordred_list_meteo[i])


'''
Ozon Werte zum jeweiligen Intervall zuordnen.
parameter: complete_data
return: complete_data(labled)
'''

def labeling(complete_data):
	for i in range(len(complete_data)):
		if complete_data[i][5] in range(0, 56):
			complete_data[i].append('Zielwert_Gruen')
			#i[len(i)-1]='Zielwert_Gruen'
		elif complete_data[i][5] in range(56,100):
			complete_data[i].append('Zielwert_Gelb')
		elif complete_data[i][5] in range(100,121):
			complete_data[i].append('Zielwert_Rot')
		elif complete_data[i][5] in range(121,141):
			complete_data[i].append('Informationsschwelle_Gruen')
		elif complete_data[i][5] in range(141,161):
			complete_data[i].append('Informationsschwelle_Gelb')
		elif complete_data[i][5] in range(161,181):
			complete_data[i].append('Informationsschwelle_Rot')
		elif complete_data[i][5] in range(181,201):
			complete_data[i].append('Alarmschwelle_Stufe1')
		elif complete_data[i][5] in range(201,221):
			complete_data[i].append('Alarmschwelle_Stufe2')
		elif complete_data[i][5] in range(221,241):
			complete_data[i].append('Alarmschwelle_Stufe3')
		else:
			complete_data[i].append('Extrem_Risiko')
	return complete_data





'''
Liste zu CSV schreiben 
'''
def write_data_to_csv(completed_list):
	df4= pd.DataFrame(completed_list,  columns=constract_header())
	df4.to_csv("./Data/enddata_labeled.csv",  encoding='utf-8')


write_data_to_csv(labeling(end_list))
