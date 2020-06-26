#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer


server = ECMWFDataServer()
    
server.retrieve({
    'stream'    : "oper",
    'levtype'   : "sfc",
    'param'     : "134.128/164.128/165.128/166.128/167.128",
    'dataset'   : "interim",
    'step'      : "0",
    'grid'      : "0.75/0.75",
    'time'      : "00/06/12/18",
    'date'      : "2017-01-01/to/2019-08-31",
    'type'      : "an",
    'class'     : "ei",
    # optionally restrict area to Europe (in N/W/S/E).
    'area'      : "75/-20/10/60",
    'target'    : "./Data/output.grib"
 })
 

