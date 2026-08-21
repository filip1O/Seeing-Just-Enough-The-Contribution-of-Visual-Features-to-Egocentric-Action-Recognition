# Human Ground Truth demo

This directory contains the response-to-response semantic-similarity demonstration used in the Human Ground Truth stage.

## Script

`SBERT_framework_response-response.py` compares every unique written action response with every response in one video-response CSV.

The script:

1. embeds complete responses with Sentence-BERT `all-mpnet-base-v2`;
2. extracts action verbs and object nouns with spaCy;
3. compares verbs and nouns with `word2vec-google-news-300`;
4. applies the study's action/object weighting and manual opposite-action rules;
5. writes a response-by-response semantic-similarity matrix.

The example input is:

```text
Human Ground Truth/example_video_responses/10788.csv
```

It contains a `Response` column plus participant/video metadata. The output is named `sem_space_10788.csv`: its first column lists comparison responses, and each remaining column contains similarities to one unique response.

## Configure before running

This is a manually configured analysis snapshot rather than a command-line program. Edit the path block near the top of the script. Because the script joins `input_path` and `target_file`, `input_path` must be a directory:

```python
input_path = "Human Ground Truth/example_video_responses"
out_path = "Human Ground Truth/example_video_responses"
target_file = "10788.csv"
filename = target_file
```

Defining `filename` ensures that out-of-vocabulary warning messages can identify the current file.

## Run

From the repository root:

```bash
python "Human Ground Truth/demos/SBERT_framework_response-response.py"
```

The first run may be slow because Sentence-BERT and the Google News Word2Vec model are downloaded and cached. Install the root `requirements.txt` first; the spaCy model `en_core_web_sm` is also required.

This demonstration constructs the semantic space for one example video. `HGT_master.xlsx` contains the study-level Human Ground Truth and recognition-consistency results.
