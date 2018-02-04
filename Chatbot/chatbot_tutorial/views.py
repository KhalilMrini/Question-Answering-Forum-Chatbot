from django.views import generic
from django.views.decorators.csrf import csrf_exempt
import json
import requests
import random
from django.utils.decorators import method_decorator
from django.http.response import HttpResponse
from django.shortcuts import render

import pandas as pd
from os import listdir
import gensim
import numpy as np
import nltk
from nltk.corpus import stopwords
import ast
stopwords = set(nltk.corpus.stopwords.words('english'))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
print('Loading word2vec...')
model = gensim.models.KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin', binary=True)
print('Loaded word2vec.')
tfp_df = pd.read_pickle('../../tfp_nonzero_df.pkl')
max_sentences = 1

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_sentence_vector(sentence):
    tokens = [token for token in nltk.word_tokenize(sentence) if token not in stopwords]
    vectors = []
    for token in tokens:
        try:
            word_vec = model.wv[token]
            vectors.append(word_vec)
        except:
            pass
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return []

def is_not_null(sent_vec):
    for element in sent_vec:
        if not element == 0.0:
            return True
    return False

def sent_to_text_similarity(sent_vec, text_vec):
    similarities = []
    for vec in text_vec:
        if is_not_null(vec):
            similarities.append(np.dot(sent_vec, vec)/(np.linalg.norm(sent_vec) * np.linalg.norm(vec)))
    if similarities:
        return np.mean(similarities)
    else:
        return np.nan

def text_to_text_similarity(sent_vecs1, sent_vecs2):
    similarities = []
    for v1 in sent_vecs1:
        if is_not_null(v1):
            similarity = sent_to_text_similarity(v1, sent_vecs2)
            if not np.isnan(similarity):
                similarities.append(similarity)
    if similarities:
        return np.mean(similarities)
    else:
        return np.nan

def text_to_corpus_similarity(text, corpus):
    sent_vecs = text_to_sent_vec(text)
    corpus_vecs = [text_to_sent_vec(other_text) for other_text in corpus]
    max_sim = 0
    index = -1
    for text_index in range(len(corpus_vecs)):
        similarity = text_to_text_similarity(sent_vecs, corpus_vecs[text_index])
        if not np.isnan(similarity) and max_sim < similarity:
            max_sim = similarity
            index = text_index
    if index >= 0:
        return corpus[index]
    else:
        return None

def compute_similarity(row, sent_vec):
    title_sim = 0
    title_word2vec = row['Title_word2vec']
    if len(title_word2vec) > 0:
        title_sim = cosine_similarity(sent_vec, title_word2vec)
    return title_sim

def compute_separate_similarity(row, sent_vecs):
    title_sim = 0
    title_word2vec = row['Title_word2vec']
    if len(title_word2vec) > 0:
        title_sim = np.dot(sent_vecs[0], title_word2vec)/(np.linalg.norm(sent_vecs[0])*np.linalg.norm(title_word2vec))
    fp_sim = text_to_text_similarity(sent_vecs[1:], row['First_Post_word2vec'])
    return title_sim + fp_sim

def compute_separate_similarity_no_question(row, sent_vecs):
    fp_sim = text_to_text_similarity(sent_vecs, row['First_Post_word2vec'])
    return fp_sim

def get_most_similar_title(sentences, sent_vecs):
    if sentences == 0:
        raise ValueError('Write something!')
    elif len(sentences) == 1:
        title_fp_sim = tfp_df.apply(lambda row: compute_similarity(row, sent_vecs[0]), axis=1)
    elif sentences[0].endswith('?'):
        title_fp_sim = tfp_df.apply(lambda row: compute_separate_similarity(row, sent_vecs), axis=1)
    else:
        title_fp_sim = tfp_df.apply(lambda row: compute_separate_similarity_no_question(row, sent_vecs), axis=1)
    return tfp_df.loc[title_fp_sim.idxmax()]

def get_response_sentences(sentences, sent_vecs, link, max_sentences):
    answer_df = pd.read_pickle('../../msg_df/msg_df_{}.pkl'.format(link))
    best_answer = answer_df.loc[answer_df[answer_df.Usefulness == max(answer_df.Usefulness)]
                                .Reply_word2vec.apply(lambda other_vecs:
                                                      text_to_text_similarity(sent_vecs, other_vecs)).idxmax()]
    best_sentence_idx = np.argmax([sent_to_text_similarity(sent_vec, sent_vecs) for sent_vec in best_answer.Reply_word2vec])
    reply_sentences = ast.literal_eval(best_answer.Reply)
    if max_sentences <= 1:
        return reply_sentences[best_sentence_idx]
    else:
        context_sent_count = int((max_sentences - 1)/2)
        sent_count = len(reply_sentences)
        lower_bound = best_sentence_idx - context_sent_count
        upper_bound = best_sentence_idx + context_sent_count + 1
        return ' '.join(reply_sentences[max(0, lower_bound - max(0, upper_bound - sent_count)):
                                        min(upper_bound + max(0, 0 - lower_bound) + ((max_sentences - 1) % 2), sent_count)])

def chatbot_answer(question, max_sentences=1):
    sentences = tokenizer.tokenize(question)
    sent_vecs = [get_sentence_vector(sent) for sent in sentences]
    most_similar_title = get_most_similar_title(sentences, sent_vecs)
    return get_response_sentences(sentences, sent_vecs, most_similar_title.Link, max_sentences)

def chat(request):
    context = {}
    return render(request, 'chatbot_tutorial/chatbot.html', context)


def respond_to_websockets(message):
    global max_sentences
    result_message = {
        'type': 'text'
    }
    if 'answer me in ' in message['text'].lower():
        global max_sentences
        max_sentences = int(''.join([char for char in message['text'] if char.isdigit()]))
        result_message['text'] = 'OK, I will now answer you in a maximum of {} sentences.'.format(max_sentences)
    else:
        result_message['text'] = chatbot_answer(message['text'], max_sentences=max_sentences)

    return result_message
