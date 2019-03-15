import numpy as np
import pandas as pd
import time
import os
import datetime
import tensorflow as tf
from math import log
from tqdm import tqdm
from rnn import RNN
from argparse import ArgumentParser
from cleaner import Cleaner
from helpers import one_hot_encode, split_data, class_distribution_analysis, batch_iterator, confusion_matrix, read_data
tqdm.pandas(tqdm())

parser = ArgumentParser()

parser.add_argument("-i", "--inputfile", dest="input_file",
                    help="Source Data file", default = "sample.csv")
parser.add_argument("-c", "--colname", dest="colname",
                    help="column name containing the text to predict on", default = "question_text")
parser.add_argument("-l", "--labels", dest="label_colname",
                    help="(Optional) column name containing the labels to predict against", default = None)
parser.add_argument("-t", "--text", dest="text",
                    help="a text line to predict on", default = None)
parser.add_argument("-d", "--checkpointdir", dest="checkpoint_dir",
                    help="relative path to chcekpoint directory", default = "runs/1552079883/checkpoints")
parser.add_argument("-b", "--batchsize", dest="batch_size",
                    help="ratio for splitting data in to train and validation set", default = 2)

args = parser.parse_args()

def evaluate(X,colname,batch_size,checkpoint_dir,labels=None,allow_soft_placement=True, log_device_placement = False):
    text_path = os.path.join(checkpoint_dir, "..", "text_vocab")
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)
    X = [str(x) for x in X]
    x_eval = np.array(list(text_vocab_processor.transform(X)))
    if labels is not None:
        classes = len(labels[0])
        y_eval = np.argmax(labels, axis=1)
    else:
        y_eval = None
        classes = None

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/logits").outputs[0]
            # Generate batches for one epoch
            iterator = batch_iterator(x_eval,y_eval, batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for item in iterator:
                x = item[0]
                batch_predictions = sess.run(predictions, {input_text: x,
                                                           dropout_keep_prob: 1.0})
                print(batch_predictions.shape)  
                print(batch_predictions[0])           
                all_predictions = np.concatenate([all_predictions, batch_predictions])
            
            all_predictions = [one_hot_encode(classes,int(pred)) for pred in all_predictions]
            print("predictions\n",all_predictions)
            if labels is not None:
                c,f = confusion_matrix(labels,all_predictions,classes)
                print("fscore ",f)
                print("confusion_matrix:")
                print(c)
                all_predictions,c,f
            return all_predictions

def prepare_data(input_file,colname,label_colname = None):
    if label_colname is None:
        X = read_data(input_file,colname)
    else:
        X,Y = read_data(input_file,colname,label_colname)
    #c = Cleaner()
    print("starting to clean data..")
    #X = [c.full_clean(text) for text in X]
    X = [text for text in X]
    if label_colname is None:
        return X
    return X,Y

def prepare_line(text):
    c = Cleaner()
    return c.full_clean(text)


if __name__ == "__main__" :
    if args.text:
        colname = 'clean_text'
        batch_size = 1
        X = [prepare_line(args.text)]
        evaluate(X,colname,batch_size,args.checkpoint_dir)
    elif args.input_file is not None and args.colname is not None:
        Y=None
        if args.label_colname is None:
            X = prepare_data(args.input_file,args.colname)
        else:
            X,Y = prepare_data(args.input_file,args.colname,args.label_colname)
        evaluate(X,args.colname,args.batch_size,args.checkpoint_dir,Y)
    else:
        print("Please specify some arguments, use --help or -h for details")
