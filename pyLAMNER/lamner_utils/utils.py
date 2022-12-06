import torch
from rouge import FilesRouge
import torch.nn as nn
import numpy as np
import random
import bleu
import json

def init_weights(m):
  for name, param in m.named_parameters():
    if 'weight' in name:
      nn.init.normal_(param.data, mean=0, std=0.01)
    else:
      nn.init.constant_(param.data, 0)

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def write_files(p, t, epoch, output_dir='predictions/', test=False, Warmup=False):
  predicted_file_name = f"{output_dir}predictions.out-"+str(epoch)+".txt"
  ref_file_name = f"{output_dir}trgs.given-"+str(epoch)+".txt"
  
  if test:

    predicted_file_name = f"{output_dir}test-predictions.out.txt"
    ref_file_name = f"{output_dir}test-trgs.given.txt"
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
  
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")

  elif Warmup:
    predicted_file_name = f"{output_dir}warm-predictions.out-"+str(epoch)+".txt"
    ref_file_name = f"{output_dir}warm-trgs.given-"+str(epoch)+".txt"
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
  
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")


  else:
    with open(predicted_file_name, "w", encoding="utf-8") as f:
      for i in p:
        f.write(i+"\n")
    
    with open(ref_file_name, "w", encoding="utf-8") as f:
      for i in t:
        f.write(i+"\n")

class Example(object):
    """A single training/test example."""
    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line=line.strip()
            js=json.loads(line)
            if 'idx' not in js:
                js['idx']=idx
            code=' '.join(js['code_tokens']).replace('\n',' ')
            code=' '.join(code.strip().split())
            nl=' '.join(js['docstring_tokens']).replace('\n','')
            nl=' '.join(nl.strip().split())            
            examples.append(
                Example(
                        idx = idx,
                        source=code,
                        target = nl,
                        ) 
            )
    return examples

def calculate_bleu(epoch,data_path,output_dir, p, test=False):
  if test:

    predicted_file_name = f"{output_dir}test.out"
    ref_file_name = f"{output_dir}test.gold"
    eval_examples = read_examples(f'{data_path}test.jsonl')
  
  else:
    predicted_file_name = f"{output_dir}val.out"
    ref_file_name = f"{output_dir}val.gold"
    eval_examples = read_examples(f'{data_path}valid.jsonl')

  predictions = []
  with open(predicted_file_name,'w') as f, open(ref_file_name,'w') as f1:
      for ref,gold in zip(p,eval_examples):
          predictions.append(str(gold.idx)+'\t'+ref)
          f.write(str(gold.idx)+'\t'+ref+'\n')
          f1.write(str(gold.idx)+'\t'+gold.target+'\n')     

  (goldMap, predictionMap) = bleu.computeMaps(predictions, ref_file_name) 
  dev_bleu=round(bleu.bleuFromMaps(goldMap, predictionMap)[0],2)
  print("  %s = %s "%("bleu-4",str(dev_bleu)))
  print("  "+"*"*20) 
  return dev_bleu

def calculate_rouge(epoch,output_dir='predictions/', test=False, Warmup=False):

  if test:

    predicted_file_name = f"{output_dir}test-predictions.out.txt"
    ref_file_name = f"{output_dir}test-trgs.given.txt"
    
    
  elif Warmup:
    predicted_file_name = f"{output_dir}warm-predictions.out-"+str(epoch)+".txt"
    ref_file_name = f"{output_dir}warm-trgs.given-"+str(epoch)+".txt"
  
  else:
    predicted_file_name = f"{output_dir}predictions.out-"+str(epoch)+".txt"
    ref_file_name = f"{output_dir}trgs.given-"+str(epoch)+".txt"

  
   
  files_rouge = FilesRouge()
  rouge = files_rouge.get_scores(
          hyp_path=predicted_file_name, ref_path=ref_file_name, avg=True, ignore_empty=True)
  return round(rouge['rouge-l']["f"]*100, 2)

def get_max_lens(train_data, test_data, valid_data, code=True):
  
  encoder_max = -1

  if code:
    for i in range(len(train_data)):
      if encoder_max< len(vars(train_data.examples[i])["code"]):
        encoder_max = len(vars(train_data.examples[i])["code"])

    for i in range(len(test_data)):
      if encoder_max< len(vars(test_data.examples[i])["code"]):
        encoder_max = len(vars(test_data.examples[i])["code"])

    for i in range(len(valid_data)):
      if encoder_max< len(vars(valid_data.examples[i])["code"]):
        encoder_max = len(vars(valid_data.examples[i])["code"])

  else:
    for i in range(len(train_data)):
      if encoder_max< len(vars(train_data.examples[i])["summary"]):
        encoder_max = len(vars(train_data.examples[i])["summary"])

    for i in range(len(test_data)):
      if encoder_max< len(vars(test_data.examples[i])["summary"]):
        encoder_max = len(vars(test_data.examples[i])["summary"])

    for i in range(len(valid_data)):
      if encoder_max< len(vars(valid_data.examples[i])["summary"]):
        encoder_max = len(vars(valid_data.examples[i])["summary"])
  return encoder_max

def print_log(text):
  with open("log.txt", "a") as f:
    f.write(text+"\n")
  return

def set_seed(SEED=1234):
  random.seed(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True