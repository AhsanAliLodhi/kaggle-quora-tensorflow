
# Sole Author and contributor of this project is Ahsan Ali Lodhi (Immatriculation Number: 03692459)

# NOTE: DOWNLOAD DATASET AND EMBEDDINGS FROM HERE for some things to work
COMPLETE DATA SET IS AVAILABLE ON : https://www.kaggle.com/c/quora-insincere-questions-classification


# Quora-Insincere-Questions-Classification---tensorflow
A Tensorflow implementation to tackle the Insincere Questions Classification problem on Kaggle

## ONE PLEA

I didn't get the time to add lines for NLTK downloads that might be needed, however,nltk provides you with the code automatically if it can't find some components, so please just bare with me and install those via command prompt. Sorry for the inconvienece.

## Dataset

We are given a training csv with 3 columns and over a million rows. Biggest problem in the dataset is class unbalance. We have approximately 93.5% data for one class and only 6.5% for the intrested class.

I tried tackling this problem via introdcuing class weights, sampling and depending on fscore instead of accuracy.

I am only submitting 100,000 rows of actual data in this submission.

COMPLETE DATA SET IS AVAILABLE ON : https://www.kaggle.com/c/quora-insincere-questions-classification

It can be downloaded and used right away if put in this project's directory.
 
### Columns:

Qid : Unique identifier for a question
question_text: Raw text of the question
target: target class the question belongs to, 1 for Insincere and 0 for sincere

## Solution

In order to solve this problem, I've tackled following things.

## 1 - Data Preprocessing
One can use preprocess.py to preprocess a csv and generate clean data.

Example Use : preprocess --inputfile myfile.csv --textcolumn target

I've performed following preprocess steps in order.

1 - Try to decode text in correct encoding.
2 - Use of ` instead of ' as apostrophe was a common error in the text, we had to correct it.
3 - use a ML model to predict word spelling corrections such as "iam sad" -> "i am sad" or "hvae yuo eataen food" -> "Have you eaten food"
4- We use a smart model which uses google's word2vec to predict contractions such as "I've" to "I have"
5 - Then we ignore every other symbol except words
6 - We remove common stop words however, with an exception of negation words such as not, no and never. Idea was that while words like "The" and "So" do not help in our task of binary classification, negation words on the other hand while being common do make a lot of sense to keep because of their impact on meaning of a sentence.
7 - Lemmatize the words.

Please refer to Cleaner.py for more details.



## 2 - Rnn dynamic model generation

After one had generate clean data via preprocess.py, next task is to train, for structure of project and mechanisms such as saving and loading models, I've taken inspiration for structure of following gidhub project. https://github.com/roomylee/rnn-text-classification-tf.

However, the idea for introducing class_weigths to tackle the problem at hand of unbalanced classes, introducing multiple stacked units for a more complex model, batch wise evaluation of validation data to facilitate validation on large datasets we are all mine. 

You can use train.py directly with its default options and it is ready to train on the data present in train.csv using column named clean_question_text for training text and target as true labels.

Example use:

python train.py

train.py provides several customizable options. Please look in to train.py --help or -h for more customizations.

Some interesting options are follwing:

--cell_type ["gru","lstm","vanilla"]  we can choose the cells of our RNN to be any of these three

--embedding ["word2vec","glove",None]  we can choose any of these three to be our pretrained word embeddings, code to accomodate more embeddings can be added later

train.py allows you to generate your dynamic model of any cell and any size using any words embedding on the fly.

## 3 - Evaluation

For evaluations, of new text on trained models I've wrote predict.py.
This file simply loads the models we created during training and provides predictions on the text we provide via its arguments.

There are three ways to do predictions.

1 - python predict.py --text "I am a sentece, tell me if i am inscensiere"
2 - python predict.py --input_file file.csv --colname textcol
3 - python predict.py --input_file file.csv --colname textcol --label_colname target

In the third case you will be provided with a fscore and accuracy while in second case just the prediction vector.

## Tip for easy use

For ease of testing, I've set up default values for both train.py and predict.py. Which means if run form where they are right now and without any changes to other files you should be able to execute following.

python preprocess.py  ==> without any other options provided will perform cleaning on train.csv and create a new file named clean_train.csv with a new column named clean_question_text
python train.py ==> without any other options create a triple stacked gru rnn with 128 hidden layers and train it on the clean data made in previous step

python predict.py ==> will load model number 1552079883 in the run folder and use that to make predictions for file sample.csv

## Final words

While I've used this code to solve one specific problem, it is generic enough to solve a lot of NLP classification tasks easily without any change in code.

Other Salient Mentions:

1 - Generic code for multiclass fscore calculation and confusion matrix, For more details look into function confusion_matrix in helpers.py
2 - Generic code for class imbalance visualization and proposal of class weights based on statistical analysis. For more details look into function class_distribution_analysis in helpers.py
3 - Cleaner.py is a reusable module to clean any text and combines very sophisticated cleanings such as spell correction and smart contraction expansion all with a super easy to use interface.