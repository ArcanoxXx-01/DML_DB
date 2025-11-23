from app.utils.csv_handler import save_dataset
from app.config.manager import DATASETS
import os, pandas as pd



save_dataset(
    path= os.path.join(DATASETS, "12345"),
    data=pd.DataFrame(pd.read_csv("./prueba.csv"))
)