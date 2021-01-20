import transformers

DEVICE = "cpu"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = "../input/uncased_L-2_H-128_A-2/"
MODEL_PATH = "../model_file/"
TRAINING_FILE = "../input/IMDB_dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
        BERT_PATH,
        do_lower_case=True
)
