# consume-corilga-api
End-point of **CORILGA** API-REST.

## Introduction
Proyecto para consumir datos pacientes alojados en la base de datos del proyecto CORILGA

## Installation
```bash
git clone https://github.com/JMasr/corilga_api.git
cd corilga_api
conda env create -f environment.yml
pip install -r requirements.txt
nano .env.keycloak # Adding the keycloak credentials
```

## Usage

### Check for new data and download it
```bash
python main.py
```
### Arguments
```bash
usage: main.py [-o] [----data_path path/to/new_data]
```


