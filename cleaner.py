import numpy as np
import pandas as pd
import re
import os
import copy
from symspellpy.symspellpy import SymSpell, Verbosity  
from pycontractions import Contractions
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
from nltk.corpus import stopwords
import codecs
import unidecode
from argparse import ArgumentParser

class Cleaner:

    def __init__(self,
                embedding_for_smart_contraction="GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin",
                spell_dictonarypath = "frequency_dictionary_en_82_765.txt"):
        self.embedding_for_smart_contraction = embedding_for_smart_contraction
        self.spell_dictonarypath = spell_dictonarypath
        self.initialized = False 


    def initialize(self):
        print("Initializing Text Cleaner..")
       
        print("Initializing Smart Contractions Module..")
        self.cont = Contractions(self.embedding_for_smart_contraction)
        self.cont.load_models()
        
        print("Initializing Stopwords Module..")
        self.stop_words = set(stopwords.words('english'))
        stop_words_without_negation = copy.deepcopy(self.stop_words)
        stop_words_without_negation.remove('no')
        stop_words_without_negation.remove('nor')
        stop_words_without_negation.remove('not')
        self.stop_words_without_negation = stop_words_without_negation
        self.pos_tags_set_1 = {'NNP'}

        print("Initializing Wordnet Lemmatizer Module..")
        self.wnl = WordNetLemmatizer()
        
        print("Initializing Spellcheck Module..")
        max_edit_distance_dictionary = 2
        prefix_length = 7
        self.sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        dictionary_path = os.path.abspath('')+"\\"+self.spell_dictonarypath
        self.sym_spell.load_dictionary(dictionary_path, 0, 1)
        
        print("Initialization complete!")

    def expand_contractions(self,text):
        try:
            text = list(self.cont.expand_texts([text], precise=False))[0]
        except Exception as e:
            return text
        return text

    
    def apostrophe_correction(self,text):
        text = re.sub("’", "'", text)
        return text
    
    
    def try_decode(self,text):
        try:
            text = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
        except:
            text = unidecode.unidecode(text)
        return text

    
    def tokenize_and_keep_only_words(self,text):
        text = re.findall(r"[a-zA-Z]+", text.lower())
        return text
    
    
    def remove_stop_words(self,text):
        text = [word for word in text if (word not in self.stop_words_without_negation and len(word)>2)]
        return text

    
    def lemmatize(self,text):
        text = [self.wnl.lemmatize(word) for word in text]
        return text

    
    def spell_check(self,text,max_edit_distance_lookup = 2):
        # tokenize each word
        text = word_tokenize(text)
        # apply pos to each word
        text = pos_tag(text)
        correct_text = []
        # for each word in sentece
        for word in text:
            # if word is not a noun
            if word[1] not in self.pos_tags_set_1:
                # check if we can correct it, then correct it
                suggestions = self.sym_spell.lookup(word[0],Verbosity.CLOSEST,
                                    max_edit_distance_lookup)
                for suggestion in suggestions:
                    # take the first correction
                    correct_text.append(suggestion.term)
                    break
            else:
                correct_text.append(word[0])
        text = ' '.join([word for word in correct_text])
        return text


    def full_clean(self,text,debug=False):
        if not self.initialized:
            self.initialize()
            self.initialized = True
        if debug:
            print("pre-clean: ",text)
        text = self.try_decode(text)
        text = self.apostrophe_correction(text)
        text = self.spell_check(text)
        text = self.expand_contractions(text)
        text = self.tokenize_and_keep_only_words(text)
        text = self.remove_stop_words(text)
        text = self.lemmatize(text)

        text = ' '.join(text)
        if debug:
            print("post-clean: ",text)
        return text




if __name__ == "__main__" :
    parser = ArgumentParser()

    parser.add_argument("-t", "--text", dest="text",
                        help="text to clean", default = "provide some text to clean")
    parser.add_argument("-d", "--debug", dest="debug",
                        help="flag to run in debug mode, which is more verbose", default = False)

    args = parser.parse_args()
    c = Cleaner()
    print(c.full_clean(args.text,args.debug))
