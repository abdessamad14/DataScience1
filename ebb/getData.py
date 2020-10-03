#Author: Abdessamad Abba

# -*- coding: utf-8 -*-
from selenium import webdriver
import shutil


driver = webdriver.Chrome()
driver.get(
    'https://www.umweltbundesamt.de/api/air_data/v2/airquality/csv?date_from=2016-12-31&time_from=24&date_to=2019-08-31&time_to=24&station=633&lang=de')
    
shutil.copy2("/home/abdessamad/Downloads/Luftqualitaet_DEHE005_Frankfurt-Hoechst_2016-12-31_23-2019-08-31_23.csv", "./Data/chemische_Daten.csv")

driver.close()


