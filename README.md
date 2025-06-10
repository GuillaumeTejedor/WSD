# Generative Weak Supervised Distance

To use these knwf files, please install KNIME version 5.4.2.
Once installed, to be able to run python code:
- Install "KNIME Python Integration" extension on KNIME software.
- Install Python version: 3.11.1
- Install necessary librairies: snorkel, scikit-learn, umap-learn, rdp, scikit-learn-extra, lifelines, numpy, pandas, scipy, scikit-learn-extra, itertools

We recommend you to link KNIME to Anaconda in order to create conda environments and
install necessary librairies directly from Anaconda software.

When opening KNIME, choose as working directory the folder "WSD" that you have downloaded from git.

There are 3 knwf files:

1) data_preparation.knwf : Used to receive raw data and to prepare it
2) stratification_concurrents_comparison.knwf : Used to cluster patients
3) clusters_prediction.knwf : Used to predict clusters of patients from clinical and biological data

You have to import them into knime in order to use them.
You have to execute them in the order.
