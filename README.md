# DML_DB

Database Sistem of DML( distributed Machine Learning models training system)

Start:

```batch
cd app
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host "0.0.0.0" --port 8000
```

Run Tests (in a new terminal)

```batch
python3 tests.py
```



docker build -t dml_db_image .
docker run -d --name dml_db_container --network dml-network --network-alias db -p 8000:8000 dml_db_image
docker run -d --name dml_db_container2 --network dml-network --network-alias db -p 7999:8000 dml_db_image
docker run -d --name dml_db_container3 --network dml-network --network-alias db -p 7998:8000 dml_db_image
docker run -d --name dml_db_container4 --network dml-network --network-alias db -p 7997:8000 dml_db_image
docker run -d --name dml_db_container5 --network dml-network --network-alias db -p 7996:8000 dml_db_image
