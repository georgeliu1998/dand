# Wrangle OpenStreetMap Data

OpenStreetMap(OSM) is an open map database built by volunteers around the world. The data created are available for free use under 
the Open Database License. As the data were crowd-sourced, there can be inconsistency in how the data is presented. In this project, 
we will improve the data quality with data wrangling and store the cleaned data in MongoDB. In particular, XML map data will be parsed, 
audited and processed for street type inconsistency before being reshaped and converted into JSON format data, later imported into
the MongoDB database.

## Requirements
The ipynb file contains the code and report for the project. To run it, you need to have Jupyter Notebook which can be obtained as part 
of [Anaconda](https://www.continuum.io/downloads). Also, you need to have [MongoDB](https://docs.mongodb.com/v3.0/installation/) installed. 
Finally, the following Python packages are needed:

- xml.etree.ElementTree
- collections 
- re
- pprint
- codecs
- json
- pymongo 
