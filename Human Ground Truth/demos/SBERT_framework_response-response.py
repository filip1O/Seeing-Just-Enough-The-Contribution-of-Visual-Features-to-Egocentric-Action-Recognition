import os
import pandas as pd
import spacy
import gensim.downloader as api
from sentence_transformers import SentenceTransformer, util
import time

# initiate S-BERT
model_s = SentenceTransformer('all-mpnet-base-v2')

# initiate word2vec
model_w = api.load('word2vec-google-news-300')

# initiate spacy
nlp = spacy.load("en_core_web_sm")

# define spacy for verbs
def find_first_verb(text):
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "VERB":
            return token.text
        else:
            act_words_resp = text.split()
            return act_words_resp[0]

# set penalty and bonus constants
p = 0.375
b = 1.375

# set directories
folder_path = "Human Ground Truth/example_video_responses/10788.csv"      # Replace with path to your downloaded file with written responses
out_folder_path = ".../..."                                               # Replace with path for saving the results

# --------- specify one file to process ---------
target_file = "10788.csv"
file_path = os.path.join(folder_path, target_file)
out_file = "sem_space_" + target_file
out_file_path = os.path.join(out_folder_path, out_file)

# ---------- PROCESS SINGLE FILE ----------
file_start_time = time.time()
v_data = pd.read_csv(file_path)
out_data = pd.DataFrame()
comp_resp_written = 0
list_calc_resp = []

# loop through rows1
for _index1, row1 in v_data.iterrows():
    curr_response1 = row1["Response"]
    curr_label = row1["Response"]
    if curr_response1 not in list_calc_resp:
        list_sem_sim = []
        list_comp_resp = []
        list_calc_resp.append(curr_response1)
        response1_emb = model_s.encode(curr_response1)

        # extract last noun from resp1
        resp1_noun_pres = 0
        so_resp1 = nlp(curr_response1)
        for r1_np in so_resp1.noun_chunks:
            resp1_noun_pres = 1
            if resp1_noun_pres == 1:
                obj_resp1 = str(r1_np).split()[-1]
        if resp1_noun_pres == 0:
            obj_resp1 = curr_response1.split()[-1]

        # loop through rows2 to compare
        for _index2, row2 in v_data.iterrows():
            curr_response2 = row2["Response"]
            list_comp_resp.append(curr_response2)
            response2_emb = model_s.encode(curr_response2)
            cos_sim = util.cos_sim(response1_emb, response2_emb).item()

            # extract last noun from resp2
            resp2_noun_pres = 0
            so_resp2 = nlp(curr_response2)
            for r2_np in so_resp2.noun_chunks:
                resp2_noun_pres = 1
                if resp2_noun_pres == 1:
                    obj_resp2 = str(r2_np).split()[-1]
            if resp2_noun_pres == 0:
                obj_resp2 = curr_response2.split()[-1]

            # object similarity
            if obj_resp1 in model_w.key_to_index and obj_resp2 in model_w.key_to_index:
                obj_similarity = model_w.similarity(obj_resp1, obj_resp2)
            else:
                print(f"'{obj_resp1}' or '{obj_resp2}' not in word2vec. ID = {curr_label}. File = {target_file}")
                obj_similarity = 1

            # find verbs
            act_resp1 = find_first_verb(curr_response1)
            act_resp2 = find_first_verb(curr_response2)

            # add cos_sim of actions manually to distinguish hidden differences
            if (act_resp1 in ("open", "opens", "opening", "opened") and act_resp2 in ("close", "closes", "closing", "closed")) or (act_resp2 in ("open", "opens", "opening", "opened") and act_resp1 in ("close", "closes", "closing", "closed")) or (act_resp1 in ("pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed", "put", "puts", "putting", "hold", "holds", "held", "holding") and act_resp2 in ("move", "moves", "moving", "moved")) or (act_resp1 in ("move", "moves", "moving", "moved") and act_resp2 in ("pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed", "put", "puts", "putting", "hold", "holds", "held", "holding")) or (act_resp1 in ("put", "puts", "putting") and act_resp2 in ("pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed", "move", "moves", "moving", "moved", "hold", "holds", "held", "holding")) or (act_resp1 in ("hold", "holds", "held", "holding") and act_resp2 in ("pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed", "move", "moves", "moving", "moved", "put", "puts", "putting")) or (act_resp2 in ("hold", "holds", "held", "holding") and act_resp1 in ("pick", "picks", "picking", "picked", "take", "takes", "taking", "took", "grab", "grabs", "grabbing", "grabbed", "move", "moves", "moving", "moved", "put", "puts", "putting")):
                 act_similarity = 0.1

            elif (" in " in curr_response1 and " out " in curr_response2) or (" out " in curr_response1 and " in " in curr_response2) or (" up " in curr_response1 and " down " in curr_response2) or (" down " in curr_response1 and " up " in curr_response2) or (" on " in curr_response1 and " off " in curr_response2) or (" off " in curr_response1 and " on " in curr_response2) or (" insert " in curr_response1 and " out " in curr_response2) or (" out " in curr_response1 and " insert " in curr_response2) or (" in " in curr_response1 and " outside " in curr_response2) or (" outside " in curr_response1 and " in " in curr_response2) or (" inside " in curr_response1 and " outside " in curr_response2) or (" outside " in curr_response1 and " inside " in curr_response2) or (" open " in curr_response1 and " on " in curr_response2) or (" on " in curr_response1 and " open " in curr_response2) or (" opens " in curr_response1 and " on " in curr_response2) or (" on " in curr_response1 and " opens " in curr_response2) or (" opening " in curr_response1 and " on " in curr_response2) or (" on " in curr_response1 and " opening " in curr_response2) or (" opened " in curr_response1 and " on " in curr_response2) or (" on " in curr_response1 and " opened " in curr_response2) or (" close " in curr_response1 and " off " in curr_response2) or (" off " in curr_response1 and " close " in curr_response2) or (" closes " in curr_response1 and " off " in curr_response2) or (" off " in curr_response1 and " closes " in curr_response2) or (" closing " in curr_response1 and " off " in curr_response2) or (" off " in curr_response1 and " closing " in curr_response2) or (" closed " in curr_response1 and " off " in curr_response2) or (" off " in curr_response1 and " closed " in curr_response2):
                act_similarity = 0.1

            else:
                # calculate cos_sim of embs for actions from response and HGT
                if act_resp1 in model_w.key_to_index and act_resp2 in model_w.key_to_index:
                    act_similarity = model_w.similarity(act_resp1, act_resp2)
                else:
                    print (f"'{act_resp1}' or '{act_resp2}' not in word2vec. ID = {curr_label}. File = {filename}")
                    act_similarity = 1 # this value is arbitrary, as the words were not in Word2Vec vocabulary and require rewording and recomputing
            
            # calculate final semantic similarity of the two responses
            sem_sim = cos_sim - (obj_similarity * p) ** 2 + (act_similarity * b) ** 2
            
            # further hidden difference that fails to correct using the previous approach
            if (curr_response1 in ("open drawer", "opens drawer", "opening drawer", "opened drawer") and curr_response2 in ("close drawer", "closes drawer", "closing drawer", "closed drawer")) or (curr_response1 in ("close drawer", "closes drawer", "closing drawer", "closed drawer") and curr_response2 in ("open drawer", "opens drawer", "opening drawer", "opened drawer")):
                sem_sim = 0.5

            list_sem_sim.append(sem_sim)

        # merge results
        if comp_resp_written == 0:
            out_data["response"] = list_comp_resp
        out_data[curr_label] = list_sem_sim
        comp_resp_written = 1

# write out
out_data.to_csv(out_file_path, index=False)

# report time
file_processing_time = time.time() - file_start_time

print(f"Processed {target_file} in {file_processing_time:.2f} seconds.")
