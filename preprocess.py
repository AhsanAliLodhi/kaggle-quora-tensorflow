import numpy as np
import pandas as pd
import re
import os
from symspellpy.symspellpy import SymSpell, Verbosity  
from pycontractions import Contractions
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords
from argparse import ArgumentParser
from tqdm import tqdm
from cleaner import Cleaner
tqdm.pandas(tqdm())


def generate_clean_data(data,colname,save_fname=None):
    c = Cleaner()
    print("starting to clean data..")
    data["clean_"+colname] = data[colname].progress_apply(c.full_clean)
    print("cleaning data complete")
    if save_fname is not None:
        data.to_csv(save_fname)
        print("data saved in ",save_fname," in current directory")
    return data




if __name__ == "__main__" :
    parser = ArgumentParser()

    parser.add_argument("-i", "--inputfile", dest="input_file",
                        help="Source Data file", default = "train.csv")
    parser.add_argument("-t", "--textcolumn", dest="text_col",
                        help="name of column containing text to clean", default = "question_text")

    args = parser.parse_args()
    data = pd.read_csv(args.input_file)
    generate_clean_data(data,args.text_col,save_fname = ("clean_"+args.input_file))

