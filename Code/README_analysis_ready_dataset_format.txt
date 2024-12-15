Below is the format and explination of the data in analysis_ready_dataset.csv

Year = The year the mask is from 
BDANumber = The number of the bda being looked at. 
            Numbers increase from North to south along the path of the river. BDA 11 does not exist. 
ReferenceNumber = The number of the reference line being looked at. 
Downstream = Each BDA is looked at 5m upstream and 5m downstream. If downstream = 0, 
            then the data is for 5m upstream of the BDA. else the data is for 5 meters downstream of the BDA. 
Comments = Comments about the row.
BDAPresent = Weather during the year, there was a BDA present at that location. 1 means a BDA is present.
LineLength = River width at that location. 