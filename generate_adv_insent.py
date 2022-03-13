import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
import json
import logging
from fp16 import FP16_Module
import GPUtil
from collections import OrderedDict
from settings import args, MODEL_CLASS, TOKENIZER, SPECIAL_TOKEN_IDS, init_logging
from settings import MEMORY_FACTOR, LEN_FACTOR, TASK_DICT, MODEL_CONFIG, DATA_ATTRS, SPECIAL_TOKENS, CONFIG_CLASS, CONFIG_NAME
from utils import QADataset, top_k_top_p_filtering, create_dataloader, logits_to_tokens, get_model_dir
from utils import sample_sequence, remove_id, get_gen_token, lll_unbound_setting
from metrics import compute_metrics
logger = logging.getLogger(__name__)
import numpy as np

# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch.set_printoptions(edgeitems=1024,linewidth=160)

import OpenAttack
import datasets


sst_suffix = [318,   428,  2423,  4633,   393,  3967,  5633, 50257]
ag_suffix  = [1148,   428,  6827,  2159,    11,  7092,    11,  7320,    11,   393, 10286,    14, 17760,    30, 50257]
snli_suffix = [532,   532, 39793,   434,   837, 8500,   837,   393, 25741,  5633, 50257]

class HookCloser:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
    
    def __call__(self, module, input_, output_):
        self.model_wrapper.curr_embedding = output_[0]

def snli_ph(premise, hypothesis):
    return "premise : \" " + premise + " \" hyperthesis : \" " + hypothesis + " \""

class MyClassifier(OpenAttack.Classifier):
    def __init__(self, model, s_tokens):
        self.model = model
        self.s_tokens = s_tokens
        self.curr_embedding = None
        self.model.transformer.wte.weight.requires_grad = True
        self.hook = self.model.transformer.wte.register_backward_hook( HookCloser(self) )
        self.loss_fct = nn.CrossEntropyLoss(reduction='mean')

    def snli_update_premise(self, premise):
        self.premise = premise

    def get_prob(self, input_):
        encd = []
        max_len = -1
        for sent in input_:
            sent = sent.replace("<unk>", "__unk__").replace("<UNK>", "__unk__")
            if args.tasks[0] == "snli":
                sent = snli_ph(self.premise, sent)
            now_encoded = TOKENIZER.encode(sent)
            if args.tasks[0] == "sst": now_encoded.extend(sst_suffix)
            if args.tasks[0] == "ag":  now_encoded.extend(ag_suffix)
            if args.tasks[0] == "snli": now_encoded.extend(snli_suffix)
            max_len = max(max_len, len(now_encoded))
            encd.append(now_encoded)
        
        len_cqs = []
        for ii in range(len(encd)):
            now_len = len(encd[ii])
            len_cqs.append(now_len)
            for jj in range(max_len-now_len): encd[ii].append(50258)

        now_cqs = torch.tensor(encd)
        len_cqs = torch.tensor(len_cqs)
        n_inputs = now_cqs.shape[0]
        
        now_bs = 10000 // now_cqs.shape[1]
        tot_iter = (now_cqs.shape[0] // now_bs) + 1
        score_ = None
        for ind in range(tot_iter):
            st = ind*now_bs
            en = min([(ind+1)*now_bs, now_cqs.shape[0]])
            past = self.s_tokens.unsqueeze(0).expand(en-st, -1, -1).cuda()
            bsz, seqlen, _ = past.shape
            past = past.view(bsz, seqlen, MODEL_CONFIG.n_layer * 2,
                             MODEL_CONFIG.n_head, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head)#.type(torch.half)
            past = past.permute([2, 0, 3, 1, 4]).split(2)
            all_outputs = self.model(input_ids=now_cqs[st:en,:].cuda(), past=past)
            outputs = all_outputs[0]
            next_logits = outputs[range(en-st), len_cqs[st:en]-1, :] / args.temperature_qa
            if args.tasks[0] == "sst":
                score_ = F.softmax(torch.cat([next_logits[:,4633].unsqueeze(0), next_logits[:,3967].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()
            if args.tasks[0] == "ag":
                if score_ is None:
                    score_ = F.softmax(torch.cat([next_logits[:,2159].unsqueeze(0), next_logits[:,7092].unsqueeze(0), next_logits[:,7320].unsqueeze(0), next_logits[:,10286].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()
                else:
                    logger.info("now_cqs.shape: [{}, {}], now ind: {}".format(now_cqs.shape[0], now_cqs.shape[1], ind))
                    score_ = np.vstack([score_, F.softmax(torch.cat([next_logits[:,2159].unsqueeze(0), next_logits[:,7092].unsqueeze(0), next_logits[:,7320].unsqueeze(0), next_logits[:,10286].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()])
            if args.tasks[0] == "snli":
                if score_ is None:
                    score_ = F.softmax(torch.cat([next_logits[:,39793].unsqueeze(0), next_logits[:,8500].unsqueeze(0), next_logits[:,25741].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()
                else:
                    logger.info("now_cqs.shape: [{}, {}], now ind: {}".format(now_cqs.shape[0], now_cqs.shape[1], ind))
                    score_ = np.vstack([score_, F.softmax(torch.cat([next_logits[:,39793].unsqueeze(0), next_logits[:,8500].unsqueeze(0), next_logits[:,25741].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()])

        return score_

    def get_grad(self, input_, labels):
        encd = []
        max_len = -1
        for sent in input_:
            sent = sent.replace("<unk>", "__unk__").replace("<UNK>", "__unk__")
            if args.tasks[0] == "snli":
                sent = snli_ph(self.premise, sent)
            now_encoded = TOKENIZER.encode(sent)
            if args.tasks[0] == "sst": now_encoded.extend(sst_suffix)
            if args.tasks[0] == "ag":  now_encoded.extend(ag_suffix)
            if args.tasks[0] == "snli": now_encoded.extend(snli_suffix)
            max_len = max(max_len, len(now_encoded))
            encd.append(now_encoded)
        
        len_cqs = []
        for ii in range(len(encd)):
            now_len = len(encd[ii])
            len_cqs.append(now_len)
            for jj in range(max_len-now_len): encd[ii].append(50258)

        now_cqs = torch.tensor(encd)
        len_cqs = torch.tensor(len_cqs)
        n_inputs = now_cqs.shape[0]

        now_bs = 10000 // now_cqs.shape[1]
        tot_iter = (now_cqs.shape[0] // now_bs) + 1
        score_ = None
        result_grad = None
        for ind in range(tot_iter):
            st = ind*now_bs
            en = min([(ind+1)*now_bs, now_cqs.shape[0]])
            past = self.s_tokens.unsqueeze(0).expand(en-st, -1, -1).cuda()
            bsz, seqlen, _ = past.shape
            past = past.view(bsz, seqlen, MODEL_CONFIG.n_layer * 2,
                             MODEL_CONFIG.n_head, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head)#.type(torch.half)
            past = past.permute([2, 0, 3, 1, 4]).split(2)

            with torch.enable_grad():
                all_outputs = self.model(input_ids=now_cqs[st:en].cuda(), past=past)
                outputs = all_outputs[0]
                next_logits = outputs[range(en-st), len_cqs[st:en]-1, :] / args.temperature_qa
                if args.tasks[0] == "sst":
                    score_ = F.softmax(torch.cat([next_logits[:,4633].unsqueeze(0), next_logits[:,3967].unsqueeze(0)]), dim=0).transpose(0,1).detach().cpu().numpy()
                    loss = self.loss_fct(torch.cat([next_logits[:,4633].unsqueeze(0), next_logits[:,3967].unsqueeze(0)]).transpose(0,1), torch.tensor(labels).cuda())
                if args.tasks[0] == "ag":
                    loss = self.loss_fct(torch.cat([next_logits[:,2159].unsqueeze(0), next_logits[:,7092].unsqueeze(0), next_logits[:,7320].unsqueeze(0), next_logits[:,10286].unsqueeze(0)]).transpose(0,1), torch.tensor(labels).cuda())
                    if score_ is None:
                        score_ = F.softmax(torch.cat([next_logits[:,2159].unsqueeze(0), next_logits[:,7092].unsqueeze(0), next_logits[:,7320].unsqueeze(0), next_logits[:,10286].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()
                    else:
                        logger.info("now_cqs.shape: [{}, {}], now ind: {}".format(now_cqs.shape[0], now_cqs.shape[1], ind))
                        score_ = np.vstack([score_, F.softmax(torch.cat([next_logits[:,2159].unsqueeze(0), next_logits[:,7092].unsqueeze(0), next_logits[:,7320].unsqueeze(0), next_logits[:,10286].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()])

                if args.tasks[0] == "snli":
                    loss = self.loss_fct(torch.cat([next_logits[:,39793].unsqueeze(0), next_logits[:,8500].unsqueeze(0), next_logits[:,25741].unsqueeze(0)]).transpose(0,1), torch.tensor(labels).cuda())
                    if score_ is None:
                        score_ = F.softmax(torch.cat([next_logits[:,39793].unsqueeze(0), next_logits[:,8500].unsqueeze(0), next_logits[:,25741].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()
                    else:
                        logger.info("now_cqs.shape: [{}, {}], now ind: {}".format(now_cqs.shape[0], now_cqs.shape[1], ind))
                        score_ = np.vstack([score_, F.softmax(torch.cat([next_logits[:,39793].unsqueeze(0), next_logits[:,8500].unsqueeze(0), next_logits[:,25741].unsqueeze(0)]), dim=0).transpose(0,1).cpu().numpy()])

                self.model.zero_grad()
                loss.backward()

                if result_grad is None:
                    result_grad = self.curr_embedding.clone().detach().cpu().numpy()
                else:
                    result_grad = np.vstack([result_grad, self.curr_embedding.clone().detach().cpu().numpy()])

                self.curr_embedding = None

        return score_, result_grad   

def dataset_mapping(x):
    if args.tasks[0] == "sst":
        return {
            "x": x["sentence"],
            "y": 1 if x["label"] > 0.5 else 0,
        }
    if args.tasks[0] == "ag":
        return {
            "x": x["text"],
            "y": x["label"],
        }
    if args.tasks[0] == "snli":
        return {
            "y": x["label"],
            "premise": x["premise"].lower(),
            "x": x["hypothesis"].lower()
        }

def test_one_to_one(task_load, task_eval, model, score_dict, s_tokens):

#    if task_eval == "sst": dataset = datasets.load_dataset("sst", split="validation").map(function=dataset_mapping)
    if task_eval == "sst": dataset = datasets.load_dataset("sst", split="test").map(function=dataset_mapping)
    if task_eval == "ag":  dataset = datasets.load_dataset("ag_news", split="test").map(function=dataset_mapping)
    if task_eval == "snli":  dataset = datasets.load_dataset("snli", split="test").map(function=dataset_mapping)

    clsf = MyClassifier(model=model, s_tokens=s_tokens)
    if args.attack == "pwws" : attacker = OpenAttack.attackers.PWWSAttacker()
    if args.attack == "scpn" : attacker = OpenAttack.attackers.SCPNAttacker()
    if args.attack == "viper": attacker = OpenAttack.attackers.VIPERAttacker()
    if args.attack == "bug"  : attacker = OpenAttack.attackers.TextBuggerAttacker()
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf)

    if task_eval == "snli":
        premise=[]
        for data in dataset: premise.append(data["premise"])
        clsf.update_premise(premise[0])

    ret = {"x_ori": [],  "y_ori": [], "pred_ori": [], "x_adv": [], "y_adv": [], "success": []}
    cnt = 0
    for data, x_adv, y_adv, info in attack_eval.eval_results(dataset):
        cnt += 1
        if cnt % 20 == 0: logger.info("now {}, total {}, percent {}%".format(cnt, len(dataset), round(cnt*100.0/len(dataset), 3)))
        if task_eval == "snli":
            ret["x_ori"].append(snli_ph(premise[cnt-1], data["x"]))
        else:
            ret["x_ori"].append(data["x"])
        ret["y_ori"].append(data["y"])
        ret["pred_ori"].append(clsf.get_pred([data["x"]])[0].item())
        if x_adv is not None:
            ret["success"].append(True)
            if task_eval == "snli":
                ret["x_adv"].append(snli_ph(premise[cnt-1], x_adv))
            else:
                ret["x_adv"].append(x_adv)
            ret["y_adv"].append(y_adv.item())
        else:
            ret["success"].append(False)
        
        if task_eval == "snli":
            if cnt != len(premise): clsf.snli_update_premise(premise[cnt])

    json.dump(ret, fp=open(task_eval + "_" + args.attack+".json", "w"))

    logger.info("finish attack")

def test_one_to_many(task_load):
    ep = args.test_ep - 1
    model_dir = get_model_dir([task_load])
    s_tokens_path = os.path.join(model_dir, "p"+str(args.preseqlen)+'lr'+str(args.learning_rate)+'model-stokens{}'.format(ep+1))
    config_path = os.path.join(model_dir,CONFIG_NAME)

    gen_token = get_gen_token(task_load)
    TOKENIZER.add_tokens([gen_token])
    SPECIAL_TOKENS[task_load] = gen_token
    SPECIAL_TOKEN_IDS[task_load] = TOKENIZER.convert_tokens_to_ids(gen_token)
    model = MODEL_CLASS.from_pretrained('../gpt2-medium-pretrained/').cuda()
    model.resize_token_embeddings(len(TOKENIZER))

    s_tokens = torch.load(s_tokens_path).cpu().to("cuda")#.cuda()
    model.ep = ep
    model.model_dir = model_dir
    logger.info("task: {}, epoch: {}".format(task_load, ep+1))
    score_dict = {k:None for k in args.tasks}
    with torch.no_grad():
        for task_eval in args.tasks:
            test_one_to_one(task_load, task_eval, model, score_dict, s_tokens)

if __name__ == '__main__':
    if args.n_gpus > 1:
        raise NotImplementedError("test can be run with only one gpu currently!")
    
    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)
    init_logging(os.path.join(args.model_dir_root, 'log_test_genadv_p{}_lr{}_{}.txt'.format(args.preseqlen, args.learning_rate, args.attack)))
    logger.info('args = {}'.format(args))

    for task_load in args.tasks:
        test_one_to_many(task_load)
