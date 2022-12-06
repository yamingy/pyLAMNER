# -*- coding: utf-8 -*-
"""CodeT5-baseline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o1v3A3OouD0yUkYx6WoWBQ3aZ5du2dFm

"""

import torch
import json
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
import math
import time
from torchtext import data
import torchtext.vocab as vocab
from lamner_utils.utils import set_seed, init_weights, print_log, get_max_lens, count_parameters, calculate_rouge, write_files, epoch_time, calculate_bleu
from src.attention import Attention
from src.encoder import Encoder
from src.decoder import Decoder
from six.moves import map
import random
from six.moves import map
from tqdm import tqdm

print(torch.__version__)
print(f"Is available: {torch.cuda.is_available()}")

try:
    print(f"Current Devices: {torch.cuda.current_device()}")
except :
    print('Current Devices: Torch is not compiled for GPU or No GPU')

print(f"No. of GPUs: {torch.cuda.device_count()}")

"""# CodeT5 Preparation
This part is mainly for preparing the vocab from pretrained codeBert model and pass it to the LAMNER framework.  **You can skip this part when training the model** **bold text**
"""

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')

#text = "def greet(user): print(f'hello <extra_id_0>!')"
#inputids = tokenizer(text, return_tensors="pt").input_ids
#model.encoder.embed_tokens(inputids)

# len(model.encoder.embed_tokens(torch.tensor([1]))[0].tolist())

# with open("data_seq2seq/codeT5_vocab.json", 'r') as f:
#  with open("data_seq2seq/codeT5_vocab.txt", "w") as fc:
#    V = json.load(f)
#    l = len(V)
#    c = 1
#    for word in V:
#      tokens_ids = [V[word]]
#      embed = model.encoder.embed_tokens(torch.tensor([1]))[0].tolist()
#      fc.write(" ".join([word] + [str(j) for j in embed]))
#      if c < l :
#        fc.write("\n")
#      c += 1
                                 
# vocab_size = len(V)

"""# CodeT5 seq2seq
where training starts
"""

CLS_INDEX = 1
EOS_INDEX = 2
PAD_INDEX = 0
UNK_INDEX = 3

class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder, src_pad_idx, device):
    super().__init__()
    
    self.encoder = encoder
    self.decoder = decoder
    self.src_pad_idx = src_pad_idx
    self.device = device
      
  def create_mask(self, src):
    mask = (src != self.src_pad_idx).permute(1, 0)
    return mask
      
  def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
      
    batch_size = src.shape[1]
    trg_len = trg.shape[0]
    trg_vocab_size = self.decoder.output_dim
    outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
    encoder_outputs, hidden = self.encoder(src, src_len)
    input = trg[0,:]
    mask = self.create_mask(src)

    for t in range(1, trg_len):
        
      output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
      outputs[t] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.argmax(1) 
      input = trg[t] if teacher_force else top1
        
    return outputs

def train(model, iterator, optimizer, criterion, clip):
  model.train()
  epoch_loss = 0
  for i, batch in enumerate(tqdm(iterator)): 
    src = batch.code
    src_len = torch.tensor([src.size(dim = 0)] * src.size(dim= 1))
    trg = batch.summary
    optimizer.zero_grad()
    output = model(src, src_len, trg)
    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    trg = trg[1:].view(-1)
    loss = criterion(output, trg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
  model.eval()
  epoch_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(iterator):
      src = batch.code
      src_len = torch.tensor([src.size(dim = 0)] * src.size(dim= 1))
      trg = batch.summary
      output = model(src, src_len, trg, 0)
      output_dim = output.shape[-1]      
      output = output[1:].view(-1, output_dim)
      trg = trg[1:].view(-1)
      loss = criterion(output, trg)
      epoch_loss += loss.item()
      
  return epoch_loss / len(iterator)

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 64):
  model.eval()
  tokens = [token.lower() for token in sentence]  
  tokens = [src_field.init_token] + tokens + [src_field.eos_token]      
  src_indexes = [src_field.vocab.stoi[token] for token in tokens]  
  src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
  src_len = torch.LongTensor([len(src_indexes)]).to(device)  
  with torch.no_grad():
    encoder_outputs, hidden = model.encoder(src_tensor, src_len)
  mask = model.create_mask(src_tensor)      
  trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
  attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)  
  for i in range(max_len):
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)            
    with torch.no_grad():
      output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
    attentions[i] = attention
    if i>=2:
      output[0][trg_indexes[-2]] = 0
    output[0][trg_indexes[-1]] = 0       
    pred_token = output.argmax(1).item()    
    trg_indexes.append(pred_token)
    if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
      break  
  trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
  
  return trg_tokens[1:], attentions[:len(trg_tokens)-1]
  
def get_preds(data, src_field, trg_field, model, device, max_len = 64):    
  trgs = []
  pred_trgs = []  
  for datum in data:
    p = ""
    t= ""
    src = vars(datum)['code']
    trg = vars(datum)['summary']    
    pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
    pred_trg = pred_trg[:-1]
    p = " ".join(pred_trg)
    p = p.strip()
    t = " ".join(trg)
    t = t.strip()
    pred_trgs.append(p)
    trgs.append(t)
      
  return pred_trgs,trgs


def translate_sentence_reps(sentence, src_field, trg_field, model, device, max_len = 64):

  model.eval()
  tokens = [token.lower() for token in sentence]
    

  tokens = [src_field.init_token] + tokens + [src_field.eos_token]
      
  src_indexes = [src_field.vocab.stoi[token] for token in tokens]
  
  src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

  src_len = torch.LongTensor([len(src_indexes)]).to(device)
  
  with torch.no_grad():
    encoder_outputs, hidden = model.encoder(src_tensor, src_len)

  mask = model.create_mask(src_tensor)
      
  trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

  attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
  
  for i in range(max_len):

    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            
    with torch.no_grad():
      output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

    attentions[i] = attention
    if i>=2:
      output[0][trg_indexes[-2]] = 0
    output[0][trg_indexes[-1]] = 0
    pred_token = output.argmax(1).item()
    
    trg_indexes.append(pred_token)

    if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
      break
  
  trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
  
  return trg_tokens[1:], attentions[:len(trg_tokens)-1]

def run_seq2seq(batch_size= 4, embedding_size= 512, hidden_dimension = 512, dropout = 0.5, epochs = 10, static = False, learning_rate = 0.1, infer = False):
  set_seed()
  CLIP = 1
  make_weights_static = static
  best_valid_loss = float('inf')
  cur_bleu = -float('inf')
  best_bleu = -float('inf')
  best_epoch = -1
  MIN_LR = 0.0000001
  MAX_VOCAB_SIZE = 50_000
  early_stop = False
  cur_lr = learning_rate
  num_of_epochs_not_improved = 0
  path = "data_seq2seq/"
  output_dir = "predictions/codet5/"


  #-------------------- OUR CODE---------------------#
  print("preparing data....")
  SRC = Field(tokenize=tokenizer.tokenize,
              init_token = tokenizer.cls_token, 
              eos_token = tokenizer.eos_token, 
              lower = False, 
              #include_lengths = True,
              fix_length = 256,
              pad_token=tokenizer.pad_token, 
              unk_token=tokenizer.unk_token)
  TRG = Field(tokenize=tokenizer.tokenize,
              init_token = tokenizer.cls_token, 
              eos_token = tokenizer.eos_token,
              lower = False,
              fix_length = 128,
              pad_token=tokenizer.pad_token, 
              unk_token=tokenizer.unk_token)
  
  train_data, valid_data, test_data = data.TabularDataset.splits(
          path=path, train='train_seq.csv',
          skip_header=True,
          validation='val_seq.csv', test='test_seq.csv', format='CSV',
          fields=[('code', SRC), ('summary', TRG)])


  # ------- TODOL how to generate codeBert_embeds ---------- #
  print("preparing vocab....")
  codeBERT = vocab.Vectors(name = path + 'codeT5_vocab.txt')
  # ------- TODOL how to generate codeBert_embeds ---------- #

  SRC.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = codeBERT
                   ) 

  TRG.build_vocab(train_data, 
                  max_size = MAX_VOCAB_SIZE 
                   )
  #-------------------- OUR CODE---------------------# 

  #*****************************************************************************************************
  print("preparing model....")
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
          (train_data, valid_data, test_data), 
          batch_size = batch_size,
          sort_within_batch = True,
          shuffle=True,
          sort_key = lambda x : len(x.code),
          device = device)
  
  
  INPUT_DIM = len(SRC.vocab)
  OUTPUT_DIM = len(TRG.vocab)
  attn = Attention(hidden_dimension, hidden_dimension)
  enc = Encoder(INPUT_DIM, embedding_size, hidden_dimension, hidden_dimension, dropout)
  dec = Decoder(OUTPUT_DIM, embedding_size, hidden_dimension, hidden_dimension, dropout, attn)
  model = Seq2Seq(enc, dec, PAD_INDEX, device).to(device)
  model.apply(init_weights)

  #*************************************************************************************
  print("Setting Embeddings")
  model.encoder.embedding.weight.data.copy_(SRC.vocab.vectors)

  #*************************************************************************************
  optimizer = optim.SGD(model.parameters(),lr=learning_rate, momentum=0.9)
  criterion = nn.CrossEntropyLoss(ignore_index = PAD_INDEX)
  cd_len = get_max_lens(train_data, test_data, valid_data, code=True)
  sm_len = get_max_lens(train_data, test_data, valid_data, code=False)
  print("Maximum Input length is " + str(cd_len) + "... Maximum Output Length is " + str(sm_len))
  print("Encoder Vocab Size " + str(INPUT_DIM) + "... Decoder Vocab Size " + str(OUTPUT_DIM))
  print("Batch Size:" + str(batch_size) + "\nEmbedding Dimension:" + str(embedding_size))
  print('The model has ' + str(count_parameters(model))+  ' trainable parameters')
  print("\nTraining Started.....")
  optimizer.param_groups[0]['lr'] = learning_rate
  
  if not(infer):
    for epoch in range(epochs):
      if MIN_LR>optimizer.param_groups[0]['lr']:
        early_stop = True
        break
  
      if num_of_epochs_not_improved==7:
        #reduce LR
        model.load_state_dict(torch.load(f'{output_dir}best-seq2seq.pt'))
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
        stepLR = optimizer.param_groups[0]['lr']
        num_of_epochs_not_improved = 0
      
      start_time = time.time()
      
      train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
      valid_loss = evaluate(model, valid_iterator, criterion)
      p, t = get_preds(valid_data, SRC, TRG, model, device)
      write_files(p,t,epoch+1, output_dir=output_dir)
      cur_bleu = calculate_bleu(epoch+1,data_path=path,output_dir=output_dir, p=p)
      torch.save(model.state_dict(), f'{output_dir}seq2seq-'+str(epoch+1)+'.pt')
  
      if best_valid_loss>valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch + 1
        num_of_epochs_not_improved = 0
      else:
        num_of_epochs_not_improved = num_of_epochs_not_improved + 1 
      
      if cur_bleu > best_bleu:
        best_bleu = cur_bleu
        torch.save(model.state_dict(), f'{output_dir}best-seq2seq.pt')
      
      if make_weights_static==True:
        model.encoder.embedding.weight.requires_grad=False
        make_weights_static=False
        print("Embeddings are static now")
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  
      print('Epoch: ' + str(epoch+1) + ' | Time: '+ str(epoch_mins) + 'm' +  str(epoch_secs) + 's')
      print('\t Learning Rate: ' + str(optimizer.param_groups[0]['lr']))
      print('\t Train Loss: ' + str(round(train_loss, 2)) + ' | Train PPL: ' + str(round(math.exp(train_loss), 2)))
      print('\t Val. Loss: ' + str(round(valid_loss, 2 )) + ' |  Val. PPL: '+ str(round(math.exp(valid_loss), 2)))
      print('\t Current Val. bleu: ' + str(cur_bleu) + ' |  Best bleu '+ str(best_bleu) + ' |  Best Epoch '+ str(best_epoch))
      print('\t Number of Epochs of no Improvement '+ str(num_of_epochs_not_improved))

  model.load_state_dict(torch.load(f'{output_dir}best-seq2seq.pt'))
  test_loss = evaluate(model, test_iterator, criterion)
  print('Test Loss: ' + str(round(test_loss, 2)) + ' | Test PPL: ' + str(round(math.exp(test_loss), 2)))
  p, t = get_preds(test_data, SRC, TRG, model, device)
  calculate_bleu(-1, data_path=path,output_dir=output_dir, p=p, test=True)
  

run_seq2seq(batch_size = 64, embedding_size=512, epochs = 20, learning_rate = 0.001)

