# DSL2018-Proj-ETH-IT
## Project description
Data Science master student semester project on the automation of answering questions received by ETH IT service using an internal FAQ list. For more details please refer to 'reports/*'

## Running the pipeline:
modify 'code/run_pipeline.py' to chosen experiment
run 'python code/run_pipeline.py' 

Project Organization
------------

    ├── README.md                 <- The top-level README for developers using this project.
    │
    ├── data                      <- Not available on Git for privacy reasons.
    │
    ├── PreparationWork           <- Jupyter notebooks. 
    │                                Contains all data cleaning and exploratory work.
    │
    ├── reports                   <- Project report, poster and presentation PDFs.
    │
    ├── code                      <- Source code for use in this project.
    │   ├── run_pipeline.py       <- Script to run the full pipeline.
    │   │                            Aknowledging bad CS practices, this is the file to modify 
    │   │                            to run different experiments
    │   │
    │   ├── embedding             <- Scripts to embedd clean data into matrix form
    │   │   ├── preprocessing.py
    │   │   └── embedding.py
    │   │
    │   ├── similarity            <- Scripts to tag ticket answers using unsupervised ML
    │   │   
    │   ├── classifier            <- Scripts to classify ticket questions using tagged FAQ using 
    │   │                            multiclass classification.
    │   │                            
    │   ├── visualisation         <- Scripts to visualise embeddings
    │   │
    │   └── archive               <- Archive of older attempts - to be ignored
    │
    └── app                       <- Final algorithm visualisation


