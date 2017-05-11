# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.
Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.
Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.
See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, json, render_template
from flask_ask import Ask, request, session, question, statement, context, audio, current_stream

import math
import os
import random
import sys
import time
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import pickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import nltk
from operator import itemgetter


model = None
vectorizer = None
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("cont_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("resp_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./Data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./Checkpoints", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
FLAGS = tf.app.flags.FLAGS

vectorizerWG = pickle.load(open("vectorizerWG.pickle", "rb"))
XWG = pickle.load(open("transformed_arr.pickle", "rb"))
freqWG = pickle.load(open("freq.pkl", "rb"))


f = open('klingon.txt', 'r')
klingon_wordslist = [l.strip() for l in f.readlines()]
f.close()

class Node:
  def __init__(self, tuple, prev_word):
    self.word = tuple[0]
    self.pos = tuple[1]
    self.next = {}
    self.prev = [prev_word]
    self.prev = set(self.prev)
    self.freq = 1

  def add_next(self, word):
    if word in self.next.keys():
      self.next[word] += 1
    else:
      self.next.update({word: 1})

  def add_prev(self, prev_word):
    self.prev.add(prev_word)
    self.prev = set(self.prev)


class Graph:
  counter = 0

  def __init__(self):
    self.word_dict = {}

  def update_word_dict(self, node):
    if node.word in self.word_dict.keys():
      node_list = self.word_dict.get(node.word)
      node_list.append(node.word)
      #print(node_list)
      self.word_dict.update({node.word: node_list})
    else:
      self.word_dict.update({node.word: [node]})

  def check_word_dict(self, word, pos):
    if word in self.word_dict.keys():
      pos_dict = self.word_dict.get(word)
      if pos in pos_dict.keys():
        return pos_dict.get(pos)
      return "No pos"
    return "No word"

  def add_sentence(self, tokens):
    tokens.insert(0, ('XSTARTX', 'X'))
    tokens.insert(len(tokens), ('XENDX', 'X'))
    prev_word = None
    prev_node = None
    for pair in tokens:
      if pair[1] == 'CD':
        continue
      curr_node = self.check_word_dict(pair[0], pair[1])
      if curr_node == "No word":
        curr_node = Node(pair, prev_word)
        self.word_dict.update({curr_node.word: {curr_node.pos: curr_node}})
      elif curr_node == "No pos":
        curr_node = Node(pair, prev_word)
        node_dict = self.word_dict.get(curr_node.word)
        node_dict.update({curr_node.pos: curr_node})
        self.word_dict.update({curr_node.word: node_dict})
      else:
        curr_node.freq += 1
        curr_node.add_prev(prev_word)
      if prev_node != None:
        prev_node.add_next(curr_node.word)
      prev_word = curr_node.word
      prev_node = curr_node

  def print_graph(self):
    for node in self.word_dict.keys():
      print("NODE " + node)
      pos_dict = self.word_dict.get(node)
      for pos in pos_dict:
        value = pos_dict.get(pos)
        print(value.word + value.pos + str(value.freq))
        print(value.prev)
        print(value.next)
          

def generate_new(tokens, sent):
  print (tokens)
  new_sents = []
  print("FUUUU")
  for i in range(0, len(tokens)-1):
    first = tokens[i]
    second = tokens[i + 1]
    first_found = Graph.check_word_dict(graph, first[0], first[1])
    second_found = Graph.check_word_dict(graph, second[0], second[1])
    if first_found != "No word" and first_found != "No pos" and second_found != "No word" and second_found != "No pos":
      first_next = first_found.next.keys()
      second_prev = second_found.prev
      options = set(first_next).intersection(second_prev)
      for reply in options:
        if (first[0] == "XSTARTX"):
          new_line = str.upper(reply) + " " + sent
          new_sents.append(new_line)
        elif (second[0] == "XENDX"):
          new_line = sent + " " + str.upper(reply)
          new_sents.append(new_line)
        else:
          # new_line = sent[0:sent.find(first[0])+len(first[0])] + " " + str.upper(reply) + " " + sent[sent.find(second[0]):]
          first_part = sent[0:sent.find(first[0])+len(first[0])] + " "
          last_part = " " + sent.replace(first_part,'')
          new_line = first_part + str.upper(reply) + last_part
          new_sents.append(new_line)
  if new_sents:        
    final_sent = rank_options(new_sents)
  else:
    final_sent = sent  
  return final_sent

def derandomizer(candidates):
  print("CANDIDATES")
  #print(candidates)
  option_file = "derandom_options"
  with open(option_file, 'w') as f:
    for s in candidates:
      f.write(s.strip()+"\n")
  f.close()
  try:
    op = subprocess.check_output("java -mx150m -cp 'stanford-parser.jar:' edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat 'semanticGraph' -printPCFGkBest 1 englishPCFG.ser.gz derandom_options", cwd="/home/ubuntu/models/tutorials/rnn/translate/stanford-parser-2008-10-26", shell=True)
    a = op.decode("utf-8").strip().replace('\r', '')
    values = a.split("\n")
    prob_list = []
    for v in values:
      idx = v.find("score ")
      prob = float(v[idx + len("score "):])
      prob_list.append(prob)
      max_prob = max(prob_list)
      options_idx = [i for i, val in enumerate(prob_list) if val == max_prob]
      final_idx = random.choice(options_idx)
      return candidates[final_idx]
  except:
    return random.choice(candidates)

def rank_options(new_sents):
  len_new_sents =  len(new_sents)
  print (len_new_sents)
  if len_new_sents > 700:
    print ("SHORTENED")
    new_sents = new_sents[:700]

  wordlist_1 = ["spock", "captain", "kirk", "doctor", "bones", "mccoy", "jim", "lieutenant", "yeoman", "janice", "uhura", "scott", "scotty", "sulu", "sir", "commodore", "commander", "chief", "picard"]
  wordlist_2 = ["space", "galaxy", "phasers", "enterprise", "janeway", "worf", "jadzia", "kasidy"]
  wordlist_3 = ["vulcan", "engineering", "control", "controls"]

  new_sents = list(set(new_sents))

  prob_list = []
  for sent in new_sents:
    words = sent.split()
    prob = 1
    log_prob = 0
    for i in range(1, len(words)):
      if words[i] not in freqWG:
        words[i] = "UNK"
      if words[i-1] not in freqWG:
        words[i-1] = "UNK"

      bigram = words[i-1] + " " + words[i]
      bi_idx = vectorizerWG.vocabulary_.get(bigram)
      idx = vectorizerWG.vocabulary_.get(words[i])
      if bi_idx != None and idx != None:
        word_count = XWG[:, idx]
        bi_count = XWG[:, bi_idx]
      else:
        word_count = 500
        bi_count = 1

      prob = prob * (bi_count / word_count)
      log_prob = log_prob + (math.log(bi_count) - math.log(word_count))

    prob_list.append(prob)
    # print(sent)
    # print(log_prob)
    
  max_prob = max(prob_list)
  # print(max_prob)
  options_idx = [i for i,val in enumerate(prob_list) if val==max_prob]
  # print (options_idx)
  # print (new_sents)
  options = [new_sents[i] for i in options_idx]
  # print(options)
  
  keywords = []
  for sent in options:
    # print ("sent: " + sent)
    key=''
    for word in word_tokenize(sent):
      if word.isupper():
        key = word
        keywords.append(key.lower())
        # print ("key "+ key)
        break     
  # print (keywords)
  
  inter = set(wordlist_1).intersection(keywords)
  if not inter:
      inter = set(wordlist_2).intersection(keywords)
  if not inter:
      inter = set(wordlist_3).intersection(keywords)
  #print(inter)
  final_idx = 0
  # No keyword matches
  if not inter:
    #final_idx = random.choice(options_idx)
    #return new_sents[final_idx]
    return derandomizer(new_sents)

  final_options = []

  for sent in options:
    #print (sent)
    wordlist = word_tokenize(sent.lower())
    for word in wordlist:
      if word in inter:
        idx = wordlist.index(word)
        # print ("idx " + str(idx))
        # print("len "+ str(len(word_tokenize(sent))))
        if idx != -1:
          if idx == 0 or idx == (len(word_tokenize(sent))-1):
            final_options.append(sent)
          else:
            print ("IN MIDDLE")
        else:
          print ("NO KEYWORD")    
  # print (final_options)

  if not final_options:
    final_options = options
    print('NOPE')
  #return random.choice(final_options)
  return derandomizer(final_options)

sess = tf.Session()

st_file = 'star_trek_dialog.bin'
# st_file = 'star_trek_dialog_subset.bin'

graph = Graph()

print ('Creating Graph')

with open(st_file, 'r') as file:
  for line in file:
    line = line.lower().strip()
    sentences = sent_tokenize(line)
    for sent in sentences:
      try:
        tokens = nltk.pos_tag([word.strip(string.punctuation) for word in sent.split(" ")])
        Graph.add_sentence(graph, tokens=tokens)
      except:
        continue
file.close()
print('Star Trek Graph formed')



def create_model(session, forward_only, path, vocab_size,domain):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      vocab_size,
      vocab_size,
      _buckets,
      FLAGS.size,
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype) 
  all_vars = tf.all_variables()
  model_vars = [k for k in all_vars if k.name.startswith(domain)]
  ckpt = tf.train.get_checkpoint_state(path) 
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path) 
    tf.train.Saver(model_vars).restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model

def load_model(sess,checkpoint,vocab_size,domain):
    # Create model and load parameters. 
    model = create_model(sess, True,checkpoint,vocab_size,domain)
    model.batch_size = 1  # We decode one sentence at a time. 

    return model

def decode(sess,model,cont_vocab,resp_vocab,sentence):
    # Decode from standard input.
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), cont_vocab)
      # Which bucket does it belong to?
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence) 

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      response = " ".join([tf.compat.as_str(resp_vocab[output]) for output in outputs])
      response = response.replace(" ' ","'")
      return response 

def predict(sent):
  global model
  global vectorizer
  if(model == None):
    with open('my_classifier.pickle','rb') as fid:
      model = pickle.load(fid)
  if(vectorizer == None):
    with open('vectorizer.pickle','rb') as fid:
      vectorizer = pickle.load(fid)
  sent = [sent]
  X_test = vectorizer.transform(sent)
  return model.predict(X_test)[0]

def load_models(sess):  
  
  with tf.variable_scope("startrek") as startrek_scope:
    startrek_model = load_model(sess,'startrek_checkpoint',40000,"startrek")

  with tf.variable_scope("cornell") as cornell_scope:
    cornell_model = load_model(sess,'cornell_checkpoint',50000,"cornell")


  # Load vocabularies.
  startrek_from_vocab_path = os.path.join(FLAGS.data_dir,
                         "startrek/vocab40000.from")
  startrek_to_vocab_path = os.path.join(FLAGS.data_dir,
                         "startrek/vocab40000.to")

  startrek_from_vocab, _ = data_utils.initialize_vocabulary(startrek_from_vocab_path)
  _, startrek_to_vocab = data_utils.initialize_vocabulary(startrek_to_vocab_path)

  cornell_from_vocab_path = os.path.join(FLAGS.data_dir,
                         "cornell/vocab50000.from")
  cornell_to_vocab_path = os.path.join(FLAGS.data_dir,
                         "cornell/vocab50000.to")

  cornell_from_vocab, _ = data_utils.initialize_vocabulary(cornell_from_vocab_path)
  _, cornell_to_vocab = data_utils.initialize_vocabulary(cornell_to_vocab_path)


  return (startrek_model,cornell_model, startrek_from_vocab, startrek_to_vocab, cornell_from_vocab, cornell_to_vocab)


startrek_model,cornell_model, startrek_from_vocab, startrek_to_vocab, cornell_from_vocab, cornell_to_vocab = load_models(sess)

def get_response_from_model(sess,sentence):
  pred_str = predict(sentence)
  pred = int(pred_str)
  response = ''
  if pred==0:
    print('Star Trek Predicted!')
    with tf.variable_scope("startrek"):
      response = decode(sess,startrek_model,startrek_from_vocab,startrek_to_vocab,sentence)
      if response == ".":
        return random.choice(klingon_wordslist)
      return response
  else:
    print('Generic Domain Predicted!')
    with tf.variable_scope("cornell"):
      response = decode(sess,cornell_model, cornell_from_vocab, cornell_to_vocab,sentence)
    if response == ".":
      return random.choice(klingon_wordslist)


    line = response.replace('-','').replace(' \' ','\'').replace(' \" ', '\"')
    #print ("STARTING WG")
    try:
      line = str(line).strip(" .")
      line = line.replace(" ' ","'")
      tokens = nltk.pos_tag([word.strip(string.punctuation) for word in line.split()])
      tokens.insert(0, ('XSTARTX', 'X'))
      tokens.insert(len(tokens), ('XENDX', 'X'))
      new_reply = generate_new(tokens, line)
    except:
      print ("EXCEPT")
      e = sys.exc_info()[0]
      print("<p>Error: %s</p>" % e)
      new_reply = random.choice(klingon_wordslist)   #TODO: SELECT FROM LIST (Klingon)

    return new_reply 

# Response when alexa skill is launched
@ask.launch
def launch():
    print ("i am here")
    stream_url = 'https://s3.amazonaws.com/cis-700-7/Star+Trek+Original+Series+Intro+(HQ)+(mp3cut.net).mp3'
    return audio().play(stream_url)

@ask.intent('AMAZON.StopIntent')
def stop():
    return audio('stopping').clear_queue(stop=True)

@ask.intent('AMAZON.PauseIntent')
def pause():
    return audio('Paused the stream.').stop()

@ask.intent('AMAZON.ResumeIntent')
def resume():
    return audio('Resuming.').resume()

# Response to any utterance to the bot (Runs the decodder of the deep neural net to get the response) 
@ask.intent("ChatIntent", convert = {"Text" : str})
def ask_intent(Text,startrek_model,cornell_model, startrek_from_vocab, startrek_to_vocab, cornell_from_vocab, cornell_to_vocab):
  Text = Text.lower()
  if not Text.endswith('.'):
    Text += '.'
  print ("INPUT: " + Text)  
  response = get_response_from_model(sess,Text)
  print ("RESPONSE" + response)
  return question(response)

if __name__ == '__main__':   
  app.run(debug=True,use_reloader=False)
