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
from helpers import one_hot_encode, split_data, class_distribution_analysis, batch_iterator, confusion_matrix, read_data
tqdm.pandas(tqdm())
parser = ArgumentParser()

parser.add_argument("-i", "--inputfile", dest="input_file",
                    help="Source Data file", default = "clean_train.csv")
parser.add_argument("-t", "--textcolumn", dest="text_col",
                    help="name of column containing text to clean", default = "question_text")
parser.add_argument("-l", "--labelcolumn", dest="label_col",
                    help="name of column containing label for each text", default = "target")
parser.add_argument("-v", "--validratio", dest="valid_ratio",
                    help="ratio for splitting data in to train and validation set", default = 0.2)
parser.add_argument("-m", "--maxlength", dest="max_sentence_length",
                    help="maximum length of a sentence allowed in words", default = 91) # because in our data, the longest sentence consists of 91 words
parser.add_argument("-z", "--sampleratio", dest="sample_percent",
                    help="lets you train on a random proportion of your given train data, can be used while developing to quick run", default = 1)  # example value 0.5 for half of data          
parser.add_argument("-w", "--classweights", dest="class_weights",
                    help="you may provide class weights in case of unbalanced classes", default = None)
parser.add_argument("-c", "--celltype", dest="cell_type",
                    help="available options for RNN type are gru, lstm, vanilla", default = "gru")
parser.add_argument("-e", "--embedding", dest="embedding",
                    help="available options for pretrained embeddings are word2vec and glove", default = "word2vec")
parser.add_argument("-f", "--embeddingpath", dest="embedding_path",
                    help="path the the embedding specified in option -e", default = "GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin")
parser.add_argument("-d", "--embeddingdim", dest="embedding_dim",
                    help="size of vector for one embedding", default = 300)
parser.add_argument("-r", "--rnnlayers", dest="rnn_layers",
                    help="Number of units of the rnn cell you want stacked up in your architechture", default = 3)      
parser.add_argument("-q", "--hiddensize", dest="hidden_size",
                    help="size of hidden layers for each Rnn unit", default = 128)
parser.add_argument("-o", "--oneminusdropout", dest="one_minus_dropout",
                    help="one minus probability of dropout", default = 0.5)
parser.add_argument("-g", "--l2reg", dest="l2_reg",
                    help="L2 regularizatoin lambda", default = 3.0)
parser.add_argument("-b", "--batchsize", dest="batch_size",
                    help="Batch size, advised to be a multiple of 32", default = 32)
parser.add_argument("-y", "--epochs", dest="epochs",
                    help="number of epochs for training", default = 3)
parser.add_argument("-j", "--learningrate", dest="learning_rate",
                    help="learning rate for training", default = 1e-3)

args = parser.parse_args()


def train(
            input_file = "clean_train.csv",
            text_col = "question_text",
            label_col = "target",
            valid_ratio = 0.2,
            max_sentence_length = 91,
            sample_percent = 1,
            class_weights = None,
            cell_type = "gru",
            embedding = "word2vec",
            embedding_path = "GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin",
            embedding_dim = 300,
            rnn_layers = 3     ,
            hidden_size = 128,
            one_minus_dropout = 0.5,
            l2_reg = 3.0,
            batch_size = 32,
            epochs = 5,
            learning_rate = 1e-3,
            allow_soft_placement =   True, 
            log_device_placement =   False, 
            display_every =   10, 
            evaluate_every =   100, 
            checkpoint_every =   100, 
            num_checkpoints =   5
        ):
    # Load and split data
    print("Loading data..")
    X,Y = read_data(input_file,text_col,label_col, sample_percent = sample_percent)

    # Create a vocanulary process
    # Its job is to assign each unique word an integer and then our sentences replace each word it's corresponding integer.
    # These mappings are later used again to substitue each word with its embedding
    # This method also trims or adds trailing zeros to padd and fit each sentence to a specific length
    print("Setting up vocabulary..")
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sentence_length)
    X = np.array(list(vocab_processor.fit_transform(X)))
    print("Vocabulary Size: ",len(vocab_processor.vocabulary_))
    num_classes =  len(Y[0])

    # split in to train and validation
    X,Y,x_val,y_val = split_data(X,Y,valid_ratio)

    # initialize tensorflow config
    print("Initializing tensorflow session..")
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print("Initializing our RNN:")
            print("\nseq_length : ",X.shape[1], 
                "\nnum_classes : ",Y.shape[1], 
                "\nvocab_size : ",len(vocab_processor.vocabulary_), 
                "\nembedding_size : ",embedding_dim,
                "\ncell_type : ",cell_type, 
                "\nhidden_size : ",hidden_size, 
                "\nl2 : ",l2_reg,
                "\nclass_weights :  ", class_weights,
                "\nbatch_size : ",batch_size,
                "\nrnn_layers :  ", rnn_layers)
            # Initiazlie our RNN 
            rnn = RNN(
                seq_length=X.shape[1], num_classes=Y.shape[1], vocab_size=len(vocab_processor.vocabulary_), embedding_size=embedding_dim,
                 cell_type=cell_type, hidden_size=hidden_size, l2=l2_reg,class_weights = class_weights,batch_size=batch_size,
                 rnn_layers = rnn_layers
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(rnn.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            val_summary_op = tf.summary.merge([loss_summary, acc_summary])
            val_summary_dir = os.path.join(out_dir, "summaries", "val")
            val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Initializing pretrained embeddings if embedding flag is up
            if embedding:
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), embedding_dim))
                
                # In case of glove, loading embedings is pretty easy
                # Just read each line, first word is the word
                # and evey thing else on the line is a vector embedding for that vector
                if "glove" in embedding:
                    with open(embedding_path, "r",encoding="utf8") as f:
                        for line in f:
                            first_word = line.partition(' ')[0]
                            rest  = line[line.index(' ') + 1:]
                            # Find if word in our vocabulary
                            idx = vocab_processor.vocabulary_.get(first_word)
                            if idx != 0:
                                # If yes then substitue the glove embedding for it instead of the random one
                                initW[idx] = np.fromstring(rest, dtype='float32',sep = " ")
                # In case of word2vec, we are given a bin file
                elif "word2vec" in embedding:
                     with open(embedding_path, "rb") as f:
                        # First line is header containing information about number of records and size of one record
                        header = f.readline()
                        vocab_size, layer1_size = map(int, header.split())
                        # Then, number of bytes in each record  = (size of a float) * size of one record
                        binary_len = np.dtype('float32').itemsize * layer1_size
                        # for each record
                        for line in range(vocab_size):
                            word = []
                            while True:
                                # Keep reading a charachter 
                                ch = f.read(1).decode('latin-1')
                                if ch == ' ':
                                    # until you find a space, then the first word is complete
                                    word = ''.join(word)
                                    break
                                if ch != '\n':
                                    word.append(ch)
                            # Try to find that first word in our vocabulary
                            idx = vocab_processor.vocabulary_.get(word)
                            if idx != 0:
                                # if found, add substitue the corespoding embedding vector with the random vector
                                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                            else:
                                f.read(binary_len)
                    
                sess.run(rnn.W_text.assign(initW))
                print("Successful to load ",embedding,"!\n")

            # Once we are done with the embeddings and basic tensorflow settings 
            # We now start with actual training routine

            # Generate batches
            itr = batch_iterator(X,Y, batch_size, epochs)
            # For each batch
            for x_batch,y_batch,start,end in itr:
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_label: y_batch,
                    rnn.keep_prob: one_minus_dropout
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % display_every== 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % evaluate_every== 0:
                    print("\nEvaluation:")
                    total_preds =  np.zeros(y_val.shape)
                    itr2 = batch_iterator(x_val,y_val, batch_size, 1,shuffle = False)
                    avg_acc = 0
                    avg_loss = 0
                    steps = 0
                    for x_eval_batch, y_eval_batch ,s,e in itr2:
                        feed_dict_val = {
                                rnn.input_text: x_eval_batch,
                                rnn.input_label: y_eval_batch,
                                rnn.keep_prob: 1.0
                            }
                        summaries_val, loss, accuracy, preds = sess.run(
                            [val_summary_op, rnn.loss, rnn.accuracy,rnn.predictions], feed_dict_val)
                        val_summary_writer.add_summary(summaries_val, step)
                        k = np.array([one_hot_encode(num_classes,label) for label in preds])
                        avg_acc += accuracy
                        avg_loss += loss
                        steps += 1
                        total_preds[s:e] = k
                    cf,f_score = confusion_matrix(y_val,total_preds,2)
                    avg_acc /= steps
                    avg_loss /= steps
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: loss {:g}, acc {:g}, fscore {:g}\n".format(time_str, avg_loss, avg_acc, f_score))
                    print("Confusion Matrix")
                    print(cf)
                # Model checkpoint
                if step % checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__" :
    train(
        input_file = args.input_file,
        text_col = args.text_col,
        label_col = args.label_col,
        valid_ratio = args.valid_ratio,
        max_sentence_length = args.max_sentence_length,
        sample_percent = args.sample_percent,
        class_weights = args.class_weights,
        cell_type = args.cell_type,
        embedding = args.embedding,
        embedding_path = args.embedding_path,
        embedding_dim = args.embedding_dim,
        rnn_layers = args.rnn_layers,
        hidden_size = args.hidden_size,
        one_minus_dropout = args.one_minus_dropout,
        l2_reg = args.l2_reg,
        batch_size = args.batch_size,
        epochs = args.epochs,
        learning_rate = args.learning_rate
        )
