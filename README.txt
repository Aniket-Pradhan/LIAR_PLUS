Bad-Fake-News
=============

NUS Selection Task
------------------

My task was to implement a model to detect/predict fake-news from the
given data-set. For this purpose, I was given the [LIAR\_PLUS
data-set](https://github.com/Tariq60/LIAR-PLUS). The LIAR\_PLUS data-set
is used for fact-checking and fake news detection released in the
[original author's paper](http://aclweb.org/anthology/W18-5513).

Setup
=====

Before running the examples, create a virtual environment, and then
install the dependencies. This would not mess with your system's local
libraries, and maintain uniformity.

    $ # setup virtualenv
    $ pip install --user virtualenv #install virtualenv (ignore if already installed)
    $ python -m virtualenv ./.venv #setup the virtualenv
    $ source ./.venv/bin/activate #activate the env

    $ (.venv) # install requirements
    $ (.venv) pip install -r requirements.txt

Training and Testing
====================

After setting up the virtual environment and the dependencies, now it is
time to train and test the model. To run the code:

    $ (.venv) # move to the scripts directory
    $ (.venv) # there is a training and testing script already made
    $ (.venv) # To run the training script:
    $ (.venv) ./train_six.sh
    $ (.venv) # It will train the model for the six-way classification.
    $ (.venv) ./test_six.sh
    $ (.venv) # It will test the model for the six-way classification.
    $ (.venv) # In order to run the model for binary classification:
    $ (.venv) ./train_binary.sh
    $ (.venv) ./test_binary.sh

By default the scripts run the Gaussian NB model. In order to run the
code for other models (Bernoulli NB, Multinomial NB or Logistic
Regression), open the corresponding scripts with a text editor, and
un-comment the model you want to train. **Do not forget to comment the
previous line.**

If you need to run the python scripts yourself, you can do that by
supplying the necessary arguments as well. As an example, in order to
view the arguments of `naive_bayes.py` write:

    $ (.venv) python naive_bayes.py -h
    usage: naive_bayes.py [-h] [--binary BOOLEAN] [--train BOOLEAN]
                          [--test BOOLEAN] [--trainfile FILE] [--testfile FILE]
                          [--model FILE] [--nb_type STRING]

    optional arguments:
      -h, --help        show this help message and exit
      --binary BOOLEAN  If you want to do a binary classification, or the default
                        six-way classification.
      --train BOOLEAN   Want to train the model? Set as True, if you want to...
      --test BOOLEAN    Want to test the model? Set as True, if you want to...
      --trainfile FILE  Specify the location of the training dataset
      --testfile FILE   Specify the location of the testing dataset
      --model FILE      Specify the location of the trained model
      --nb_type STRING  Values: GaussianNB or BernoulliNB or MultinomialNB.

Provide the necessary arguments and then you're good to go and run the
scripts.

### Another easy way to run the script?

There is a GUI for the given application as well. How to run it?

    $ (.venv) cd scripts/
    $ (.venv) python gui.py

That's it. The buttons will make it easy for you to train and test the
models on-the-go, without command-line access. Do give it a try.

TF-IDF Word Embeddings
======================

I also implemented the TF-IDF word embeddings for the current data-set.
The same can be found in `scripts/tf_idf.py`. In order to run the
script:

    $ (.venv) cd scripts/
    $ (.venv) python tf_idf.py
    usage: tf_idf.py [-h] -f FILE [-r BOOLEAN]

    optional arguments:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  Specify the location of the dataset to analyze
      -r BOOLEAN            If you want to re-iterate and generate tf-idf again.

I ran the script on the training data, and was able to generate the
embeddings after running for some time on a high performance computer
(HPC). The saved data can be found in `temp_logs/` directory. In order
to utilize the previously generated data, one can simply run the script
and set the `-r` argument as `false` (by default), since it takes very
long to process the huge training data. If you want to re-generate the
data again, set the `-r` argument as `true` As an example:

    $ (.venv) python tf_idf.py -f ../dataset/train2.tsv -r true # to regenerate the data
    $ (.venv) python tf_idf.py -f ../dataset/train2.tsv -r true # to used the saved data
