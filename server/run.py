import argparse

from fastapi import FastAPI
from .config import ModemmConfigDynamic, ModemmConfigStatic

#
parser = argparse.ArgumentParser(description='Run time config for Modemm')
parser.add_argument("-c", "--config_file", type=str, help="Server config file path", default="./config.json")
parser.add_argument("-d", "--dynamic-config", type=bool, help="Use the dynamic config loader", default=True)
parser.add_argument("-p", "--port", type=int)

args = parser.parse_args()

# dynamic reload or static?
config = ModemmConfigDynamic(args.config_file) if args.dynamic_config else ModemmConfigStatic(args.config_file)
app = FastAPI()

@app.get("/")
def home_page():
    return {"meow": "meow"}  # placeholder


@app.get("/server/config")
def get_config():
    pass
