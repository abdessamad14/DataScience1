# Author: Abdessamad Abba, Abdelkhalek Kamkoum, Adil Ennaciri, Goethe-Universitaet Frankfurt am Main, FB 12 - Informatik und Mathematik
# Matrikelnummer: 5776119, 6331906, 6324191
# Date: 26.06.2020


Ecoblackboxverzeichnis enthält folgende Elemente:

------------------------------------------------------------------------------------------------------------------------------------------------------------------
	- dataholder.sh: Agent, der für das Abholen der meteorologischen Daten zuständig ist.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
	- dataparser.py: Agent, der CSV Daten unter Data/ aufruft. Er exportiert eine CSVdatei, die meteorologischen sowie chemischen Daten enthält.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
	- dataanalyser.py: ruft enddata_labled.csv und führt due Regression sowie Klassifizierung(Training sowie Test) aus.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	- enddata_labeled.csv: Dataparserouput bzw Dataanalyserinput.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
	- chemische_Daten.csv: chemische Daten, die aus Umweltbunsamt Webseite gedownloadet werden.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
	- meteorological_data.csv: meteorologische Daten, die aus EMCWF Server gedownloadet werden.


Das Aufrufen der Anwendung:

	- Die Anwendung ist unter:
		* Betriebssystem:
			Distributor ID:	Ubuntu
			Description:	Ubuntu 18.04.2 LTS
			Release:	18.04
			Codename:	bionic
		* python:
			Python 2.7.15 :: Anaconda, Inc.
	entwickelt und getestet.

	- Das Aufrufen:
	
	1. Dataholder aufrufen:
		
		$ chmod +x dataholder.sh
		$ ./dataholder.sh
		 
	2. Dataparser aufrufen:
		$ python2.7 dataparser.py
		
	3. Dataanalyser aufrufen:
		$ python2.7 dataanalyser.py







