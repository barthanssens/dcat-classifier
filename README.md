# DCAT-classifier

Machine learning test project to automatically classify datasets according to the [13 categories](http://publications.europa.eu/mdr/resource/authority/data-theme/html/data-theme-eng.html) of the European Data Portal, using  the popular [scikit-learn](http://scikit-learn.org/stable/) Python library.

## Input
Input is the DCAT-AP XML [export file](https://github.com/Fedict/dcat) from the [Belgian open data portal](http://data.gov.be). Most of the 5000+ datasets already have one or more categories (manually) assigned to them, which should be a good start for the machine learning algorithms.

## Potential issues
The result depends on:
  - the quality of the descriptions and the relevance of the keywords (if present)
  - the correctness of the (often manually asigned) categories of the training sample
  - the number of available samples per category (the distribution varies wildly)


