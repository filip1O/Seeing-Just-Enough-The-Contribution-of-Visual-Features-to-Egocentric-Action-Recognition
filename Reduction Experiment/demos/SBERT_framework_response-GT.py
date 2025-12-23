import os
import pandas as pd
import spacy
import gensim.downloader as api
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util
import time

# initiate S-BERT
model_s = SentenceTransformer('all-mpnet-base-v2')

# initiate spacy
nlp = spacy.load("en_core_web_sm")

# define spacy for verbs
def find_first_verb(text):
    # Process the text through SpaCy NLP pipeline
    doc = nlp(text)
    # Iterate over the tokens to find the first verb
    for token in doc:
        if token.pos_ == "VERB":
            return token.text  # Return the first verb found
        else:
            act_words_resp = text.split()
            act_resp = act_words_resp[0]
            return act_resp

# initiate word2vec
model_w = api.load('word2vec-google-news-300')


# set penalty and bonus constants
p = 0.375
b = 1.375

# ---------------- PROCESS SINGLE FILE ----------------
input_path = "Reduction Experiment/example_video_responses/pooled_LL_LL_LL_03159.csv"    # Replace with path to your downloaded file with written responses to a video cropping,
                                                                                        # the result will be written back into this file
start_time = time.time()
print(f"Processing file: {filename}")

file_start_time = time.time()
p_data = pd.read_csv(input_path)

list_sem_sim = []

for _index, row in p_data.iterrows():
    curr_response = row["Response"].lower()
    curr_label = row["Response"]
    curr_GT = row["HGT"].lower()

    # embeddings
    response_emb = model_s.encode(curr_response)
    GT_emb = model_s.encode(curr_GT)
    cos_sim = util.cos_sim(response_emb, GT_emb).item()

    # last noun from response
    resp_noun_pres = 0
    so_resp = nlp(curr_response)
    for r_np in so_resp.noun_chunks:
        resp_noun_pres = 1
        span_obj_resp = r_np
        obj_resp = str(span_obj_resp).split()[-1]
        break
    if resp_noun_pres == 0:
        obj_resp = curr_response.split()[-1]

    # last noun from GT
    GT_noun_pres = 0
    so_GT = nlp(curr_GT)
    for gt_np in so_GT.noun_chunks:
        GT_noun_pres = 1
        span_obj_GT = gt_np
        obj_GT = str(span_obj_GT).split()[-1]
        break
    if GT_noun_pres == 0:
        obj_GT = curr_GT.split()[-1]

    # object similarity
    if obj_resp in model_w.key_to_index and obj_GT in model_w.key_to_index:
        obj_similarity = model_w.similarity(obj_resp, obj_GT)
    else:
        print(f"'{obj_resp}' or '{obj_GT}' not in word2vec. ID = {curr_label}. File = {filename}")
        obj_similarity = 1

    # Find action verbs and compare
    act_resp = find_first_verb(curr_response)
    act_GT = find_first_verb(curr_GT)
    # add cos_sim manually to distinguish hidden differences
    if (act_resp in ("open", "opens", "opening", "opened") and act_GT in ("close", "closes", "closing", "closed")) or (
            act_GT in ("open", "opens", "opening", "opened") and act_resp in (
    "close", "closes", "closing", "closed")) or (act_resp in (
    "pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed",
    "put", "puts", "putting", "hold", "holds", "held", "holding") and act_GT in (
                                                 "move", "moves", "moving", "moved")) or (
            act_resp in ("move", "moves", "moving", "moved") and act_GT in (
    "pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed",
    "put", "puts", "putting", "hold", "holds", "held", "holding")) or (
            act_resp in ("put", "puts", "putting") and act_GT in (
    "pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed",
    "move", "moves", "moving", "moved", "hold", "holds", "held", "holding")) or (
            act_resp in ("hold", "holds", "held", "holding") and act_GT in (
    "pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed",
    "move", "moves", "moving", "moved", "put", "puts", "putting")) or (
            act_GT in ("hold", "holds", "held", "holding") and act_resp in (
    "pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed",
    "move", "moves", "moving", "moved", "put", "puts", "putting")):
        act_similarity = 0.1
    elif (" in " in curr_response and " out " in curr_GT) or (" out " in curr_response and " in " in curr_GT) or (
            " up " in curr_response and " down " in curr_GT) or (" down " in curr_response and " up " in curr_GT) or (
            " on " in curr_response and " off " in curr_GT) or (" off " in curr_response and " on " in curr_GT) or (
            " insert " in curr_response and " out " in curr_GT) or (
            " out " in curr_response and " insert " in curr_GT) or (
            " in " in curr_response and " outside " in curr_GT) or (
            " outside " in curr_response and " in " in curr_GT) or (
            " inside " in curr_response and " outside " in curr_GT) or (
            " outside " in curr_response and " inside " in curr_GT) or (
            " open " in curr_response and " on " in curr_GT) or (" on " in curr_response and " open " in curr_GT) or (
            " opens " in curr_response and " on " in curr_GT) or (" on " in curr_response and " opens " in curr_GT) or (
            " opening " in curr_response and " on " in curr_GT) or (
            " on " in curr_response and " opening " in curr_GT) or (
            " opened " in curr_response and " on " in curr_GT) or (
            " on " in curr_response and " opened " in curr_GT) or (
            " close " in curr_response and " off " in curr_GT) or (
            " off " in curr_response and " close " in curr_GT) or (
            " closes " in curr_response and " off " in curr_GT) or (
            " off " in curr_response and " closes " in curr_GT) or (
            " closing " in curr_response and " off " in curr_GT) or (
            " off " in curr_response and " closing " in curr_GT) or (
            " closed " in curr_response and " off " in curr_GT) or (" off " in curr_response and " closed " in curr_GT):
        act_similarity = 0.1
    else:
        # calculate cos_sim of embs for actions from response and HGT
        if act_resp in model_w.key_to_index and act_GT in model_w.key_to_index:
            act_similarity = model_w.similarity(act_resp, act_GT)
        else:
            print(f"'{act_resp}' or '{act_GT}' not in word2vec. ID = {curr_label}. File = {filename}")
            act_similarity = 0
    # combined semantic similarity
    sem_sim = cos_sim - (obj_similarity * p) ** 2 + (act_similarity * b) ** 2
    
    # override if needed
    if (curr_response in ("open drawer", "opens drawer", "opening drawer", "opened drawer") and
        curr_GT in ("close drawer", "closes drawer", "closing drawer", "closed drawer")) or \
            (curr_response in ("close drawer", "closes drawer", "closing drawer", "closed drawer") and
                curr_GT in ("open drawer", "opens drawer", "opening drawer", "opened drawer")):
        sem_sim = 0.5

    list_sem_sim.append(sem_sim)

# add to dataframe
p_data['sem_sim'] = list_sem_sim

# save back
p_data.to_csv(input_path, index=False)

# timing
file_processing_time = time.time() - file_start_time

print(f"Finished processing {filename} in {file_processing_time:.2f} seconds")

