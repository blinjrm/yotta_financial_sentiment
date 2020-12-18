# financial_sentiment

Projet N°2 - Deep Learning  
Sujet : Sentiment analysis of financial news headlines  

Copyright : 2020, Damien Mellot & Jerome Blin


___

# Project Organization


    ├── README.md                      <- The top-level README for developers using this project.
    │
    ├── activate.sh                    <- Configure the environment (PYTHONPATH, dependencies, virtual environment).
    │
    ├── download_models.sh             <- Configure the environment (PYTHONPATH, dependencies, virtual environment).
    │
    ├── data
    │   ├── app                        <- Data used for the webapp.
    │   ├── training                   <- Data for training the model.
    │   ├── prediction                 <- Data to use for predictions.
    │   └── output                     <- Output of the prediction script.
    │
    ├── docs                           <- Documentation for the project
    │
    ├── logs                           <- Folder containing the logs
    │
    ├── model                          <- Folder containing the trained model, to be used for predictions. 
    │
    ├── notebooks                      <- Jupyter notebooks.
    │
    ├── pyproject.toml                 <- Poetry file with dependencies.
    │
    ├── Makefile                       <- Commands to launch scripts.
    │
    └── src                            <- Source code for use in this project.
        ├── __init__.py                <- Makes src a Python module.
        │
        ├── interface                  <- Scripts to create the webapp.
        │   └── app.py
        │
        ├── application                <- Scripts to use the trained models to make predictions.
        │   ├── train.py
        │   ├── predict_string.py
        │   ├── predict.py
        │   └── scraping.py
        │
        ├── domain                     <- Sripts to fine-tune and save the model.
        │   ├── app_utils.py
        │   ├── predict_utils.py
        │   └── train_utils.py
        │
        ├── infrastructure             <- Scripts to load the raw data in a Pandas DataFrame.
        │   └── infra.py
        │
        └── settings                   <- Scripts containing the settings.
            └── base.py

___

# Getting Started

## 1. Clone this repository

```
$ git clone <this project>
$ cd <this project>
```

## 2. Setup your environment

Goal :   
Add the directory to the PYTHONPATH  
Install the dependencies (if some are missing)  
Create a local virtual environment in the folder `./.venv/` (if it does not exist already)  
Activate the virtual environment  

- First: check your python3 version:

    ```
    $ python3 --version
    # examples of outputs:
    Python 3.6.2 :: Anaconda, Inc.
    Python 3.7.2

    $ which python3
    /Users/benjamin/anaconda3/bin/python3
    /usr/bin/python3
    ```

    - If you don't have python3 and you are working on your mac: install it from [python.org](https://www.python.org/downloads/)
    - If you don't have python3 and are working on an ubuntu-like system: install from package manager:

        ```
        $ apt-get update
        $ apt-get -y install python3 python3-pip python3-venv
        ```

- Now that python3 is installed create and configure your environment:

    ```
    $ make init
    ```
    
   This command will run the script activate.sh, and: 
    - Add the project directory to your PYTHONPATH
    - Install the requiered dependencies
    - Create (if necessary) the virtual environmnet
    - Activate the virtual environment

    You sould **always** use this command when working on the project in a new session. 


## 3. Train (fine-tune) the model

- The labelled data to train the model on must be in the data/training/ directory
- Run the training script using : 

    *from the shell :*
    ```
    $ python src/domain/train.py training_data.csv
    $ python src/domain/train.py training_data.csv --epochs=5
    ```

    *from a python interpreter :*
    ```
    >>> run src/domain/train.py training_data.csv
    >>> run src/domain/train.py training_data.csv --epochs=5
    ```

   Note that you must pass the filename containing the training data as an argument. The file must be of type .csv.  
   The optinal argument you can pass are :  
    - filename (string): File containing the data - must be of type csv.    
    - path (string, optional): Path where filename is located. Defaults to stg.TRAINING_DATA_DIR.  
    - headline_col (string, optional): Name of the columns containing the headlines. Defaults to stg.HEADLINE_COL.  
    - sentiment_col (string, optional): Name of the columns containing the target (sentiment). Defaults to stg.SENTIMENT_COL.  
    - epochs (int, optional): Number of epochs for fine-tuning. Defaults to 5.  
   
- The trained model, tokenizer and config will be saved in the model/ directory. 

*Alternatively, you can download the models we fine-tuned (RoBERTa-base and distilRoBERTa-base) from a cloud storage bucket, using the command:*

    ```
    $ make init
    ```
*The two models will be downloaded to the model/ directory and can be used for financial sentiment classification.*

## 4. Scrape new data

Some headlines have already been scraped, covering a period from mars to novembre 2020. This data can be found under data/prediction/, since it will be passed to the model for sentiment analysis. 
- If you want to scrape additional data, you can run the scraping srcipt using: 

    *from the shell :*
    ```
    $ python src/application/scraping.py
    $ python src/application/scraping.py --website="reuters" --early_date="2020-09-01"
    ```

    *from a python interpreter :*
    ```
    >>> run src/application/scraping.py
    >>> run src/application/scraping.py --website="reuters" --early_date="2020-09-01"
    ```

- The file containing the predictions will be saved in the data/prediction/ directory as a csv file named *scraped_headlines.csv*  

## 5. Use the model for predictions

- The data used for prediction must be in the data/prediction/ directory
- Run the classification script using: 

    *from the shell :*
    ```
    $ python src/application/predict.py data_4_prediction.csv
    ```

    *from a python interpreter :*
    ```
    >>> run src/application/predict.py data_4_prediction.csv
    ```

- The file containing the predictions will be saved in the data/outputs/ directory as a csv file named *data_with_predictions.csv*  


## 6. Test the model on a single sentiment analysis

You can also test the model by classifiying a single sentence   
- Run the classification script using: 

    *from the shell :*
    ```
    $ python src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"
    ```

    *from a python interpreter :*
    ```
    >>> run src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"
    ```

- The label and score will be printed to the terminal.  

## 7. Launch the web app

You can use our model and review our results using a web app. 
- Launch the streamlit app using: 

    *from the shell :*
    ```
    $ streamlit run src/interface/app.py
    ```

- Copy the url and open it in a web browser
 