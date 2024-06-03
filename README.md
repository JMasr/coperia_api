# Consume-FHIR-API-of-COPERIA
This repository provides an implementation for consuming an API using the [FHIR (Fast Healthcare Interoperability Resources)](https://hl7.org/fhir/) standard for medical data. The project is developed by the [Multimedia Technologies Group](https://gtm.uvigo.es/en/) at the **atlanTTic Research Center, Universidade de Vigo**, in collaboration with [Bahia Software](https://bahiasoftware.es/home) and the [Galcian Health Service (SERGAS)](https://www.sergas.es/).
Proyecto para consumir datos pacientes alojados en la base de datos del proyecto CORILGA

## About

To address the significant impact of Post-Acute Sequelae of SARS-CoV-2 (PASC) on global health, the [COPERIA project](https://coperia.es/) has been initiated in collaboration with the "Persistent COVID Unit of the Ourense Hospital" and primary care centers in the health area. The main objective of this project is to develop and clinically validate a comprehensive multidisciplinary platform that utilizes artificial intelligence for the diagnosis, empowerment, and clinical management of PASC patients. As part of this broader initiative, our study specifically focuses on investigating the potential of voice signal analysis as a non-invasive and accessible tool for classifying individuals as either patients with PASC or healthy individuals. The clinical study conducted in the context of the COPERIA project received ethical approval from the Clinical Research Ethics Committee of Galicia, and all procedures were conducted in compliance with the ethical principles outlined in the Declaration of Helsinki. Informed consent was obtained from all participants prior to their involvement in the study. The study was registered in the US Clinical Trials Registry under the code [NCT05629793].

The FHIR-API-Consumer project is designed to facilitate the integration and consumption of medical data through an API that adheres to the FHIR standard. This ensures interoperability and standardized communication across different healthcare systems and applications.

## Features
* Seamless integration with FHIR-compliant APIs
* Support for retrieving and managing various FHIR resources
* Example implementations for common medical data workflows
* Easy-to-follow setup and usage instructions
* Extraction of acoustic features
* Training of AI models
* Evaluation and performance testing of the AI models

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/JMasr/corilga_api.git
```
2. Navigate to the project directory:
```bash
cd corilga_api
```
3. Import the envairoment using conda:
```bash
conda env create -f env.yml
```
4. Install requirements:
```bash
pip install -r requirements.txt
```

5. Adding the keycloak credentials
```bash
nano .env.keycloak
```

6. Check for new data and download it
```bash
python main.py
```

### Arguments
```bash
main.py [-o] [----data_path path/to/new_data]
```

## Acknowledgements
*. This work was supported by the CONECTA COVID programme, co-financed by the European Regional Development Fund (ERDF) within the Galicia ERDF operational programme 2014-2020 as part of the EU’s response to the COVID19 pandemic, and Axencia Galega de Innovacion (GAIN).

*. This work has received financial support from the Xunta de Galicia (Centro singular de investigación de Galicia accreditation 2019-2022).

*. This research has been funded by the Galician Regional Government under project ED431B 2021/24“GPC".

*. Thanks to the “Unidad de COVID Persistente del Hospital de Ourense” and the patients involved in the study.
