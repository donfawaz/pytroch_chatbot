import re
import itertools
import pickle
import random
from config import Config

import torch


PAD_token = Config.PAD_token # Used for padding short sentences
SOS_token = Config.SOS_token  # Start-of-sentence token
EOS_token = Config.EOS_token  # End-of-sentence token

MAX_LENGTH = Config.MAX_LENGTH  # Maximum sentence length to consider
MIN_COUNT = Config.MIN_COUNT    # Minimum word count threshold for trimming

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)
            
            
def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
#             print(len(input_sentence), len(output_sentence))
            if len(input_sentence) > 1 and len(output_sentence) > 1:
                keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs


class PrepareData:
    def __init__(self, corpus_name, datafile):
        self.corpus_name = corpus_name
        self.datafile = datafile
        
    # def normalize_arabic(self, text):
    #     text = re.sub("[إأآا]+", "ا", text)
    #     text = re.sub("ى", "ي", text)
    #     text = re.sub("ؤ", "ء", text)
    #     text = re.sub("ئ", "ء", text)
    #     text = re.sub("ة", "ه", text)
    #     text = re.sub("گ", "ك", text)
    #     text = re.sub(r'[^ا-ي]', ' ', text)
    #     text = re.sub(' +', ' ', text)

    #     return text

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filterPair(self, p):
        # Input sequences need to preserve the last word for EOS token
        # if len(p)==2:
        return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

    # Filter pairs using filterPair condition
    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    # Using the functions defined above, return a populated voc object and pairs list
    def loadPrepareData(self):
        print("Start preparing training data ...")
        voc, pairs = Voc(self.corpus_name), self.datafile
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)
        return voc, pairs
    
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.tensor(mask, dtype=torch.bool)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        # print("Input sentence-",pair[0])
        # print("Output sentece-",pair[1])
        # print("-----------------------")
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

# if __name__ == '__main__':
#     path = '/Users/fawazalqaoud/Documents/project/python/arabic_chatbot/data/arabic_q_a.pkl'
#     # dataset = []
#     with open(path, 'rb') as f:
#         lines = pickle.load(f)

#     data = PrepareData('chatbot', lines)
#     voc, pairs = data.loadPrepareData()
    
#     small_batch_size = 5
#     batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
#     input_variable, lengths, target_variable, mask, max_target_len = batches

#     print("input_variable:", input_variable)
#     print("input shape:", input_variable.shape)
#     print("lengths:", lengths)
#     print("target_variable:", target_variable)
#     print("Output shape:",target_variable.shape)
#     print("mask:", mask)
#     print("max_target_len:", max_target_len)