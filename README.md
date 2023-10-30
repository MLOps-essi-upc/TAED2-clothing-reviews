TAED2 - Clothing Reviews
==============================

This project has been carried out within the framework of the Advanced Topics in Data Engineering 2 course with the aim of gaining experience in software engineering and software quality for data science and ML projects, learning to build ML model components following software engineering practices, and deploying ML model components following software engineering practices.

Our deep learning project focuses on predicting product quality based on customer reviews in the women's e-commerce sector. Leveraging advanced machine learning techniques, our model assesses the reviews submitted by customers to determine if a product meets high standards. Through sentiment analysis, it evaluates the sentiment, content, and context of the reviews to make predictions. This project serves as a valuable tool for businesses and consumers alike, streamlining product quality assessments and improving customer experiences.

This repository contains a comprehensive set of scripts, data, and detailed instructions to reproduce both the MLOps pipeline and the project results. By following the provided steps, you can recreate the entire machine learning and operationalization process, ensuring transparency and reproducibility in our work.

For more information, check the [Dataset Card](https://github.com/MLOps-essi-upc/TAED2-clothing-reviews/blob/main/datasetcard.md) and the [Model Card](https://github.com/MLOps-essi-upc/TAED2-clothing-reviews/blob/main/modelcard.md). 

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── .dvc
    │   ├── .gitignore
    │   └── config
    ├── .dvcignore
    ├── .gitignore
    ├── .pylintrc
    ├── config
    │   └── kaggle_connection_config.json   <- Configuration file for Kaggle API connection.
    ├── data
    │   ├── .gitignore
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   │   ├── .gitignore
    │   │   ├── train 
    │   │   │   └── train_data.csv.dvc 
    │   │   └── test
    │   │       └── test_data.csv.dvc 
    │   └── raw            <- The original, immutable data dump.
    │       └── raw_data.csv.dvc
    │
    ├── datasetcard.md     <- Dataset card containing dataset information.
    ├── dvc.lock 
    ├── dvc.yaml
    ├── fastapi_nginx_template
    │
    ├── gx              <- Great Expectations configuration and tests folder.
    │   ├── .gitignore
    │   ├── checkpoints
    │   │   └── reviews_checkpoint.yml
    │   ├── expectations
    │   │   └── .ge_store_backend_id
    │   │   └── reviews_training_suite.json
    │   ├── plugins/custom_data_docs/styles
    │   │   └── data_docs_custom_styles.css
    │   └── great_expectations.yml
    │
    ├── metrics         <- Metrics and emissions folder.
    │   ├── emissions.csv
    │   └── scores.json
    │
    ├── model             <- Trained and serialized model.
    │   ├── .gitignore
    │   └── transfer-learning.pt.dvc
    │
    ├── modelcard.md      <- Model card containing model information.
    │
    ├── notebooks          <- Jupyter notebook.
    │   └── exploratory_analysis.ipynb
    │
    ├── params.yaml
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── app         <- FastAPI app directory.
    │   │   ├── __init__.py
    │   │   ├── api.py 
    │   │   └── schemas.py
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── __init__.py
    │   │   ├── get_and_save_data.py
    │   │   ├── preprocess_data.py
    │   │   └── process_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   ├── __init__.py
    │   │   ├── extract.py
    │   │   ├── prepare.py
    │   │   └── validate.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── old_models
    │   │   │   ├── __init__.py
    │   │   │   ├── model_lstm.py
    │   │   │   ├── model_random_forest.py
    │   │   │   └── model_svc.py
    │   │   ├── __init__.py
    │   │   ├── model_pipeline.py
    │   │   ├── test_model.py
    │   │   └── train_model.py
    │   │
    │   ├── __init__.py 
    │   │
    │   └── tests  <- PyTest Testing Scripts
    │       ├── __init__.py
    │       ├── test_api.py
    │       ├── test_get_and_save_data.py
    │       ├── test_preprocess_data.py
    │       ├── test_process_data.py
    │       ├── test_test_model.py
    │       └── test_train_model.py
    │
    ├── test_environment.py
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
