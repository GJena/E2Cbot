from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask, json, render_template
from flask_ask import Ask, request, session, question, statement, context, audio, current_stream

import math, pdb
import os
import random
import sys
import time
import logging
import pickle
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import nltk, requests

model = None
vectorizer = None

app = Flask(__name__)
ask = Ask(app, "/")
logging.getLogger("flask_ask").setLevel(logging.DEBUG)

vectorizerWG = pickle.load(open("/home/gjena/nov_18/vectorizerWG.pickle", "rb"))
XWG = pickle.load(open("/home/gjena/nov_18/transformed_arr_new.pickle", "rb"))
freqWG = pickle.load(open("freq_new.pickle", "rb"))

f = open('klingon.txt', 'r')
klingon_phrases = [l.strip() for l in f.readlines()]
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
            # print(node_list)
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
    # print(tokens)
    generated_sent = []
    for i in range(0, len(tokens) - 1):
        first = tokens[i]
        second = tokens[i + 1]
        first_found = Graph.check_word_dict(graph, first[0], first[1])
        second_found = Graph.check_word_dict(graph, second[0], second[1])
        if first_found != "No word" and first_found != "No pos" and second_found != "No word" and second_found != "No pos":
            first_next = first_found.next.keys()
            second_prev = second_found.prev
            options = set(first_next).intersection(second_prev)
            for reply in options:
                if first[0] == "XSTARTX":
                    new_line = str.upper(reply) + " " + sent
                    generated_sent.append(new_line)
                elif second[0] == "XENDX":
                    new_line = sent + " " + str.upper(reply)
                    generated_sent.append(new_line)
                else:
                    # new_line = sent[0:sent.find(first[0])+len(first[0])] + " " + str.upper(reply) + " " + sent[sent.find(second[0]):]
                    first_part = sent[0:sent.find(first[0]) + len(first[0])] + " "
                    last_part = " " + sent.replace(first_part, '')
                    new_line = first_part + str.upper(reply) + last_part
                    generated_sent.append(new_line)
    if generated_sent:
        final_sent = rank_options(generated_sent)
    else:
        final_sent = sent
    return final_sent


def derandomizer(candidates):
    #print("CANDIDATES")
    # print(candidates)
    option_file = "derandom_options"
    with open(option_file, 'w') as f:
        for s in candidates:
            f.write(s.strip() + "\n")
    f.close()
    try:
        op = subprocess.check_output(
            "java -mx150m -cp 'stanford-parser.jar:' edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat 'semanticGraph' -printPCFGkBest 1 englishPCFG.ser.gz derandom_options",
            cwd="/home/ubuntu/models/tutorials/rnn/translate/stanford-parser-2008-10-26", shell=True)
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
    # len_new_sents = len(new_sents)
    # print(len_new_sents)
    # if len_new_sents > 700:
    #     print("SHORTENED")
    #     new_sents = new_sents[:700]

    new_sents = new_sents[:500]
    wordlist_1 = ["spock", "captain", "kirk", "doctor", "bones", "mccoy", "jim", "lieutenant", "yeoman", "janice",
                  "uhura", "scott", "scotty", "sulu", "sir", "commodore", "commander", "chief", "picard"]
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
            if words[i - 1] not in freqWG:
                words[i - 1] = "UNK"

            bigram = words[i - 1] + " " + words[i]
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
    options_idx = [i for i, val in enumerate(prob_list) if val == max_prob]
    # print (options_idx)
    # print (new_sents)
    options = [new_sents[i] for i in options_idx]
    # print(options)

    keywords = []
    for sent in options:
        # print ("sent: " + sent)
        key = ''
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
    # print(inter)
    final_idx = 0
    # No keyword matches
    if not inter:
        # final_idx = random.choice(options_idx)
        # return new_sents[final_idx]
        return derandomizer(new_sents)

    final_options = []

    for sent in options:
        # print (sent)
        wordlist = word_tokenize(sent.lower())
        for word in wordlist:
            if word in inter:
                idx = wordlist.index(word)
                # print ("idx " + str(idx))
                # print("len "+ str(len(word_tokenize(sent))))
                if idx != -1:
                    if idx == 0 or idx == (len(word_tokenize(sent)) - 1):
                        final_options.append(sent)
                    else:
                        pass
                        #print("IN MIDDLE")
                else:
                    #print("NO KEYWORD")
                    # print (final_options)
                    pass
    if not final_options:
        final_options = options
        #print('NOPE')
    # return random.choice(final_options)
    return derandomizer(final_options)


# sess = tf.Session()

st_file = 'star_trek_dialog.bin'
# st_file = 'star_trek_dialog_subset.bin'

graph = Graph()

print('Creating Graph')

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


def predict_route(user_input):
    global model, vectorizer
    if model is None:
        with open('my_classifier.pickle', 'rb') as fid:
            model = pickle.load(fid, encoding='latin1')
    if vectorizer is None:
        with open('vectorizer.pickle', 'rb') as fid:
            vectorizer = pickle.load(fid, encoding='latin1')
    prediction = model.predict(vectorizer.transform([user_input]))
    return prediction


def get_response_from_model(text, model_id):
    """Fetch response from opennmt models"""
    headers = {
        'Content-Type': 'application/json',
    }
    data = [{"src": text, "id": model_id}]
    answer = requests.post('http://127.0.0.1:5000/translator/translate', headers=headers, data=json.dumps(data))
    response = json.loads(answer.content)[0][0]['tgt'].encode('ascii', 'ignore').decode('utf-8')
    if len(response) < 2: # Null or single char responses
        return random.choice(klingon_phrases)
    if model_id == 200: # Generic model used
        #print("*************Generic!!!!!!*********")
        line = response.replace('-', '').replace(' \' ', '\'').replace(' \" ', '\"')
        # print ("STARTING WG")
        try:
            line = str(line).strip(" .?")
            line = line.replace(" ' ", "'")
            #pdb.set_trace()
            #print(line)
            #print(string)
            #print(word for word in line.split())
            tokens = nltk.pos_tag([word.strip(string.punctuation) for word in line.split()])
           # print(tokens)
            tokens.insert(0, ('XSTARTX', 'X'))
            tokens.insert(len(tokens), ('XENDX', 'X'))
            #print(tokens)
            new_reply = generate_new(tokens, line)
        except:
            #print("EXCEPT")
            e = sys.exc_info()[0]
            print("<p>Error: %s</p>" % e)
            new_reply = random.choice(klingon_phrases)  # TODO: SELECT FROM LIST (Klingon)
        return new_reply
    return response


@ask.launch
def new_game():
    # stream_url = 'https://s3.amazonaws.com/cis-700-7/Star+Trek+Original+Series+Intro+(HQ)+(mp3cut.net).mp3'
    # return audio().play(stream_url)
    welcome_msg = render_template('welcome')
    return question(welcome_msg)


@ask.intent("TextIntent", convert={'text': str})
def user_text(text):
    print("INPUT: " + text)
    prediction = predict_route(text)  # Checking if Star Trek or not
    if not prediction:
        predicted = 'Star Trek'
        model_id = 100
    else:
        predicted = 'general'
        model_id = 200

    response = get_response_from_model(text, model_id)
    print("RESPONSE" + response)
    return question(response)




    # print(json.loads(answer.content))
    # print(json.loads(answer.content)[0])
    # print(json.loads(answer.content)[0][0])
    # print(json.loads(answer.content)[0][0]['tgt'])

    # print(type(response))
    # response = answer.content + "." + predicted
    # response = get_response_from_model(Text)

    # return question(text)


if __name__ == '__main__':
    app.run(port=6000, debug=True)
