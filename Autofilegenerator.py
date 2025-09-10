import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] : %(message)s')

FileList=["src/__init__.py","src/helper.py","src/prompt.py",".env","setup.py","app.py","research/trials.ipynb"]

for f in FileList:
    f=Path(f)
    fdir,fname=os.path.split(f)

    if fdir!="":
        os.makedirs(fdir, exist_ok=True)
        logging.info(f"creating directory;{fdir} for file;{fname}")
    
    if(not os.path.exists(f)) or (os.path.getsize(f)==0):
        with open(f,'w') as fp:
            pass
            logging.info(f"creating empty file;{fname}")
    else:
        logging.info(f"file;{fname} already exists")