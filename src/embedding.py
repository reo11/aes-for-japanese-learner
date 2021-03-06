import MeCab
import gensim
import numpy as np
from typing import List, Tuple
import re
import joblib
import subprocess
from tqdm import tqdm
import warnings
import sentencepiece as spm
warnings.filterwarnings('ignore')
mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/unidic -Owakati')
word2vec = gensim.models.Word2Vec.load("../trained_models/ja.bin")
ignore_list = ["@", "未定義語", "EOS"]


def create_embedding_matrix(sentences: List[str], col: str):
    word_set = set()
    word2index = {}
    index2word = {}

    word_vecs = []
    for sentence in sentences:
        words = mecab_parser(sentence)
        for word in words:
            word_set.add(word)

    matrix_len = len(word_set) + 1
    weights_matrix = np.zeros((matrix_len, 300))

    count = 0
    unk_count = 0
    print("Loading vocab")
    word2index[word] = 0
    index2word[0] = "unk"
    weights_matrix[0] = np.zeros(300)
    for i, word in enumerate(tqdm(sorted(word_set)), start=1):
        word2index[word] = i
        index2word[i] = word
        try:
            weights_matrix[i] = word2vec[word]
        except:
            weights_matrix[i] = np.zeros(300)
            unk_count += 1
        count += 1
    cov = 1 - (unk_count / count)
    print(f"Coverage: {cov:.3g}")
    joblib.dump(weights_matrix, f"../model/{col}/weights_matrix")
    joblib.dump(word2index, f"../model/{col}/word2index")
    joblib.dump(index2word, f"../model/{col}/index2word")


def mecab_parser(doc: str) -> List[str]:
    return mecab.parse(doc).split()


def jumanpp(doc: str) -> str:
    proc = subprocess.run(["echo '" + str(doc) + "' | jumanpp"],stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True)
    return proc.stdout.decode("utf8")


def juman_parser(doc: str) -> List[str]:
    result = jumanpp(doc)
    words = []
    for word in result.split("\n"):
        if word.split() == []:
            continue
        word = word.split()[0]
        if word in ignore_list:
            continue
        words.append(word)
    return words

def sp_parser(doc: str) -> List[str]:
    return sp.EncodeAsPieces(doc)
