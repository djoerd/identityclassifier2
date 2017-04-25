Twitter profile identity classifier using Tensorflow
====================================================

Adapted from MNIST example. Current implementation (wrongly) assumes 
that every profile has 1 identity.

Using data from: https://github.com/annapriante/identityclassifier

Usage:

    wget 'https://raw.githubusercontent.com/annapriante/identityclassifier/master/Train5_dataset.csv'
    wget 'https://raw.githubusercontent.com/annapriante/identityclassifier/master/Test5_dataset.csv'
    python tweets_classify.py 

