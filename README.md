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
