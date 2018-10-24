# Style Transfer with Integrated Gradients

## Basic Seq2Seq Model
 * Now working: In Seop, Youn

## Basic Seq2Seq Model 2
 * Now working: Do Hyeon, Lee

### Prepro
 * Download yelp data into `[PROJECT_DIR]/data/yelp`. We need the file `[PROJECT_DIR]/data/yelp/yelp_academic_dataset_review.json`.
 * Download yelp data into `[PROJECT_DIR]/data/glove`. We need the file `[PROJECT_DIR]/data/glove/glove.6B.100d.txt`.
 * Run the code below. It will make `train.data`, `dev.data`, `test.data` and `word2vec.data` at the `[PROJECT_DIR]/seq2seq2/data/` folder.
```
python -m seq2seq2.prepro
```
### Train
 * Run the code below. `--mode train` can be ommited while default mode is train.
```
python -m seq2seq2.main --mode train
```
### Test
 * Run the code below.
```
python -m seq2seq2.main --mode test
```
### Visualize
 * Run the code below. `--vismode` can be `ig`, `sent_len`, `word_cnt`, `sent_ig_list`, `node_ig_list`. If `--vismode` omitted, default value is `node_ig_list`.
```
python -m seq2seq2.main --mode visualize --vismode sent_ig_list
```
## DP-GAN
 * Now working: None
 * We decided to hold on until basic seq2seq model was completed.
