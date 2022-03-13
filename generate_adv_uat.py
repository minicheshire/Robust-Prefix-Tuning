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
import copy
from operator import itemgetter
from itertools import groupby

# training settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

torch.set_printoptions(edgeitems=1024,linewidth=160)

global save_curr_embedding

save_curr_embedding = None

def myhook(module, input_, output_):
    global save_curr_embedding
    save_curr_embedding = output_[0]

def test_one_to_one(task_load, task_eval, model, score_dict, s_tokens):

    logger.info("start to test { task: %s (load) %s (eval), seq train type: %s }" % (task_load, task_eval, args.seq_train_type))

    test_qadata = QADataset(TASK_DICT[task_eval]["test"] , "test", SPECIAL_TOKEN_IDS[task_load]).sort()
    test_qadata.max_a_len = 1
    max_a_len = test_qadata.max_a_len
    test_dataloader = create_dataloader(test_qadata, "test")
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))

    need_process = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [0 for _ in range(n_examples)]

    cnt = 0
    count_examples = 0
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    model.transformer.wte.weight.requires_grad = True
    model.transformer.wte.register_backward_hook(myhook)
    with torch.enable_grad():
        curr_trigger = [0]*args.uat_len
        nw_beams = [ ( curr_trigger,  0 ) ]
        best_trigger = None
        best_loss    = 0
        for epep in range(args.uat_epoch): # 5 epoch for uat
            for n_steps, (cqs, len_cqs, _, _, Y, _, _) in enumerate(test_dataloader):
                # assume n_gpus == 1
                now_cqs = cqs[0]
                len_cqs = len_cqs[0]
                n_inputs = now_cqs.shape[0]

                count_examples += now_cqs.shape[0]
                logger.info("done {}, total {}, now {}%".format(count_examples, n_examples, int(count_examples*10000.0/n_examples)/100))

                for i in range(args.uat_len):
                    beams = copy.deepcopy(nw_beams)
                    nw_beams = []
                    cnt_beam = 0
                    for trigger, _ in beams:
                        cnt_beam += 1
                        prepend = torch.tensor([trigger for _ in range(n_inputs)])
                        pen_cqs = torch.cat([prepend, now_cqs], dim=1)
                        pen_len_cqs = len_cqs + args.uat_len # shift-right

                        logger.info("\t uat_epoch: {}, now_examples: {}, uat_now_len: {}, uat_now_beam: {}".format(epep, count_examples, i, cnt_beam))

                        past = s_tokens.unsqueeze(0).expand(n_inputs, -1, -1).cuda()
                        bsz, seqlen, _ = past.shape
                        past = past.view(bsz, seqlen, MODEL_CONFIG.n_layer * 2,
                                        MODEL_CONFIG.n_head, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head)#.type(torch.half)
                        past = past.permute([2, 0, 3, 1, 4]).split(2)
                        all_outputs = model(input_ids=pen_cqs.cuda(), past=past)
                        
                        outputs = all_outputs[0]
                        if "gpt2" in args.model_name:
                            pasts = all_outputs[1]
                        next_logits = outputs[range(n_inputs), pen_len_cqs-1, :] / args.temperature_qa

                        loss = loss_fct(next_logits, Y[0][range(n_inputs),len_cqs-1].cuda())

                        del past
                        del pen_cqs
                        torch.cuda.empty_cache()

                        model.zero_grad()
                        loss.backward()
                        idx = torch.matmul(model.transformer.wte.weight[:len(TOKENIZER.encoder)], save_curr_embedding[:, i, :].mean(dim=0)).argsort(descending=True)[:5].tolist() # beamsize=5

                        for cw in idx:
                            tt = trigger[:i] + [cw] + trigger[i + 1:]

                            prepend = torch.tensor([tt for _ in range(n_inputs)])
                            pen_cqs = torch.cat([prepend, now_cqs], dim=1)
                            pen_len_cqs = len_cqs + args.uat_len ## shiftshift
                            
                            past = s_tokens.unsqueeze(0).expand(n_inputs, -1, -1).cuda()
                            bsz, seqlen, _ = past.shape
                            past = past.view(bsz, seqlen, MODEL_CONFIG.n_layer * 2,
                                            MODEL_CONFIG.n_head, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head)#.type(torch.half)
                            past = past.permute([2, 0, 3, 1, 4]).split(2)
                            all_outputs = model(input_ids=pen_cqs.cuda(), past=past)
                            
                            outputs = all_outputs[0]
                            if "gpt2" in args.model_name:
                                pasts = all_outputs[1]
                            next_logits = outputs[range(n_inputs), pen_len_cqs-1, :] / args.temperature_qa

                            loss = loss_fct(next_logits, Y[0][range(n_inputs),len_cqs-1].cuda())
                            nw_beams.append((tt, loss.item()))

                            del past
                            del pen_cqs
                            torch.cuda.empty_cache()                            
                    nw_beams = sorted(nw_beams, key=lambda x: x[1], reverse=True)
                    nw_beams = list(map(itemgetter(0), groupby(nw_beams)))[:5]

                logger.info("{} uat_epoch finished, now loss1 {}, trigger1 {}".format(epep, nw_beams[0][1], str(nw_beams[0][0])))
                logger.info("{} uat_epoch finished, now loss2 {}, trigger2 {}".format(epep, nw_beams[1][1], str(nw_beams[1][0])))
                logger.info("{} uat_epoch finished, now loss3 {}, trigger3 {}".format(epep, nw_beams[2][1], str(nw_beams[2][0])))
                logger.info("{} uat_epoch finished, now loss4 {}, trigger4 {}".format(epep, nw_beams[3][1], str(nw_beams[3][0])))
                logger.info("{} uat_epoch finished, now loss5 {}, trigger5 {}".format(epep, nw_beams[4][1], str(nw_beams[4][0])))
                if nw_beams[0][1] > best_loss:
                    best_loss = nw_beams[0][1]
                    best_trigger = nw_beams[0][0]
                logger.info("now best trigger {}, now best loss {}".format(best_trigger, best_loss))
        print(best_trigger, best_loss)

def test_one_to_many(task_load):
    score_dicts = []
    ep = args.test_ep - 1
    if True:
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
    init_logging(os.path.join(args.model_dir_root, 'log_test_genadv_p{}_lr{}_uat.txt'.format(args.preseqlen, args.learning_rate)))
    logger.info('args = {}'.format(args))

    for task_load in args.tasks:
        test_one_to_many(task_load)
