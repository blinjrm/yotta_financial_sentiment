.. financial_sentiment documentation master file, created by
   sphinx-quickstart on Tue Dec 15 20:27:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Financial_sentiment's documentation
===============================================

.. warning::

   Work in progress!!!!!!

   *Release December 21st 2020*

   **Yotta Academy** 
 

0. **Installation**
===============
===========================
1) *Clone this repository*
===========================

.. code-block:: bash

   $ git clone <this project>
   $ cd <this project>

===========================
2) *Setup your environment*
===========================

| Goal :
| Add the directory to the PYTHONPATH
| Install the dependencies (if some are missing)
| Create a local virtual environment in the folder ``./.venv/`` (if it does not exist already)
| Activate the virtual environment

   * First: check your python3 version:
   .. code-block:: bash

      $ python3 --version
      # examples of outputs:
      Python 3.6.2 :: Anaconda, Inc.
      Python 3.7.2

      $ which python3
      /Users/benjamin/anaconda3/bin/python3
      /usr/bin/python3


   If you don't have python3 and you are working on your mac: install it from http://python.org
         
   If you don't have python3 and are working on an ubuntu-like system: install from package manager:

   .. code-block:: bash

      $ apt-get update
      
      $ apt-get -y install python3 python3-pip python3-venv

   * Now that python3 is installed create and configure your environment:
   .. code-block:: bash
      
      $ source activate.sh
   
   | This command will :
      
      * Add the project directory to your PYTHONPATH
      * Install the requiered dependencies
      * Create (if necessary) the virtual environment
      * Activate the virtual environment

   | You sould always use this command when working on the project in a new session.

==============================
3. *Train (fine-tune) the model*
==============================

* The raw data to train the model on must be in the data/training/ directory

* Run the training script using :

*from the shell :*

.. code-block:: bash

   $ python src/domain/train.py training_data.csv
   $ python src/domain/train.py training_data.csv --epochs=5

*from a python interpreter :*

.. code-block:: bash

   >>> run src/domain/train.py training_data.csv
   >>> run src/domain/train.py training_data.csv --epochs=5

| Note that you must pass the filename containing the training data as an argument. The file must be of type .csv.
| The optinal argument you can pass are :
   
   * filename (string): File containing the data - must be of type csv.
   
   * path (string, optional): Path where filename is located. Defaults to stg.TRAINING_DATA_DIR.
   
   * headline_col (string, optional): Name of the columns containing the headlines. Defaults to stg.HEADLINE_COL.
   
   * sentiment_col (string, optional): Name of the columns containing the target (sentiment). Defaults to stg.SENTIMENT_COL.
   
   * epochs (int, optional): Number of epochs for fine-tuning. Defaults to 5.

* The trained model, tokenizer and config will be saved in the model/ directory.

================================
4. *Use the model for predictions*
================================

* The data used for prediction must be in the data/prediction/ directory

* Run the classification script using :

*from the shell :*

.. code-block:: bash

   $ python src/application/predict.py data_4_prediction.csv

*from a python interpreter :*

.. code-block:: bash 

   >>> run src/application/predict.py data_4_prediction.csv

================================================
5. *Test the model on a single sentiment analysis*
================================================

You can also test the model by classifiying a single sentence

* Run the classification script using :

*from the shell :*

.. code-block:: bash 

   $ python src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"

*from a python interpreter :*

.. code-block:: bash 

   >>> run src/application/predict_string.py "After a violent crash, the stock doesn't seem to recover"


* The label and score will be printed to the terminal.

.. toctree::
   :maxdepth: 2
   : caption: Contents:



