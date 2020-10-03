#Author: Abdessamad Abba

#!/bin/sh
#!/usr/bin/env python2.7

#Meteorologische Daten aus EMCWF Server aufrufen
python2.7 dataholder.py
#Waehle Frankfurt am Main als Region
grib_ls -l 50.11,8.68,1 -p dataDate,dataTime,name,shortName ./Data/output.grib > ./Data/meteorological_data.csv

#Chemische Daten aus Umweltbundesamt Webseite aufrufen
python2.7 getData.py
