## Usage  
$**python3**  detect_object.py  input-image-path  
  
### Example  
$**cd**  src  
$**python3**  detect_object.py  ../test.jpg  

### Steps to setup
* Install MongoDB v4.0.18
* Install MongoDB driver for Python
    `pip install pymongo==3.10.1`
* Dump previously prepared AWA2 data (combined_csv.csv file)  to MongoDB
    `mongoimport --db=zsl_database --collection=awa --type=csv --headerline --file=combined_csv.csv`
* Transform the data using 
    `cd  src/helpers  `
    `python3  store_data.py`
* Insert the previously generated word2vec data to the DB
    `cd  src/helpers  `
    `$python3  store_awa_vectors.py`
(Word2vec data was generated  by running 'src/helpers/extract_awa_vectors.py' code in Google Colab and data downloaded as 'data/vectors_data.json')


