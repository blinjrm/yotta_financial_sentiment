.. financial_sentiment documentation master file, created by
   sphinx-quickstart on Tue Dec 15 20:27:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Financial_sentiment's documentation
===============================================

.. warning::

   Work in progress!!!!!!

   *Release December 21th 2020*

   **Yotta Academy** 
   
   * "Code is Law"
   * "Didn't you knew that?
      1. Maybe

   1. Yes I did.
   2. No I didn't.

   ``pip install best_package_of_all_times``

Header 1
========
========
Header 2
========
--------
Header 3
--------
^^^^^^^^
Header 4
^^^^^^^^
********
Header 5
********

code sample::

   def _load_data_from_csv(self, filename, path):
        if self._check_file_extension(filename):
            df = self._open_file(filename, path)
            return df

.. code-block:: python

   def _check_file_extension(filename):
        logging.info("Confirm file extension is .csv")
        if filename.endswith(".csv"):
            return True
        else:
            logging.error("Extension must be .csv")
            raise FileExistsError("Extension must be .csv")

useful




























   


Projet N°2 - Deep Learning

Sujet : Sentiment analysis of financial news headlines

Copyright : 2020, Damien Mellot & Jerome Blin

``Tout le monde doit avoir accès aux meilleurs informations financières
en un instant``

.. toctree::
   :maxdepth: 2
   : caption: Contents:
   
   sample_doc.rst
   mardown_doc.md
   notebook.ipynb




