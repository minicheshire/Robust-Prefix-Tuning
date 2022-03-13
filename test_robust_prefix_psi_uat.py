"""
Note on the "awareness" of uat_len

During inference, we are NOT (and also should not be) aware of the length of the UAT. Otherwise, the scenario is trivial as all we have to do is shift right the input so that the trigger tokens are cut out. However, there are three times that "args.uat_len" appears in this file (Lines 90, 154, 178). We want to make it clear that each of the "args.uat_len" in this file only contributes to the updated length of the current input. In other words, we always treat "len_cqs + args.uat_len" as a whole (len_cqs denotes the lengths of the original inputs). After prepending the UATs to the original inputs, we are not aware of the original lengths any more.

As you might have found, we hard-code the exploited UATs in this file (Lines 70~87) for our proof-of-concept experiments. However, the tricky thing is that in our code, the data is allocated into mini-batches and those with shorter lengths in the batch are padded with -1 at the end of the sequence. It turns out that for each data batch, a complementary list (len_cqs) is needed to indicate the end of the real sentence (w/o padding). After prepending the inputs with UATs, we need to update the len_cqs in order to know the end position of the sentences w/ the UATs, and that's why the "args.uat_len" appears in this file.

The "len_cqs + args.uat_len" approach is equivalent to checking the position of the rightmost non-"-1" token for each datum in the batch to update the length. With this equivalence, we want to assure you that the uat_len information is not leaked during inference in our experiments.
"""
import torch
import torch.nn as nn
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

def PCA_proj(data, proj, isTestbs1=False):
    if isTestbs1:
        data_proj = torch.mm(data, proj)
    else:
        mean = data.mean(dim=0, keepdim=True)
        data = data - mean
        data_proj = torch.mm(data, proj) + mean
    return data_proj

def test_one_to_one(task_load, task_eval, model, score_dict, s_tokens, projs, PMP_lr, PMP_iter):

    logger.info("start to test { task: %s (load) %s (eval), seq train type: %s }" % (task_load, task_eval, args.seq_train_type))

    test_qadata = QADataset(TASK_DICT[task_eval]["test"] , "test", SPECIAL_TOKEN_IDS[task_load]).sort()
    max_a_len = test_qadata.max_a_len
    test_dataloader = create_dataloader(test_qadata, "test")
    n_examples = len(test_qadata)
    logger.info("len of test dataset: {}".format(n_examples))

    need_process = OrderedDict()
    qa_results = [0 for _ in range(n_examples)]
    all_pasts = [[0 for _ in range(n_examples)] for __ in range(MODEL_CONFIG.n_layer)]
    max_tot_lens = [0 for _ in range(n_examples)]

    cnt = 0
    loss_fct = nn.CrossEntropyLoss(reduction='sum')
    criterion_layer = nn.MSELoss()
    loss_history = 9999.
    count_examples = 0
    for n_steps, (cqs, len_cqs, _, _, Y, _, _) in enumerate(test_dataloader):
        # assume n_gpus == 1
        now_cqs = cqs[0]
        len_cqs = len_cqs[0]
        n_inputs = now_cqs.shape[0]

        # prepend the exploited UATs to inputs
        if task_eval == "sst":
            prepend_pos = torch.tensor([[31225, 11234, 36766] for _ in range(n_inputs)]).permute(1,0)
            prepend_neg = torch.tensor([[34531, 21275, 37928] for _ in range(n_inputs)]).permute(1,0)
            now_yy = Y[0][range(n_inputs), len_cqs-1]
            prepend = ((now_yy == 3967) * prepend_pos + (now_yy == 4633) * prepend_neg).permute(1,0)
        if task_eval == "ag":
            prepend1 = torch.tensor([[21387, 21685, 41805] for _ in range(n_inputs)]).permute(1,0)
            prepend2 = torch.tensor([[276, 43566, 21685] for _ in range(n_inputs)]).permute(1,0)
            prepend3 = torch.tensor([[21520, 25606, 15835] for _ in range(n_inputs)]).permute(1,0)
            prepend4 = torch.tensor([[25375, 14420, 22729] for _ in range(n_inputs)]).permute(1,0)
            now_yy = Y[0][range(n_inputs), len_cqs-1]
            prepend = ((now_yy == 2159) * prepend1 + (now_yy == 7092) * prepend2 + (now_yy == 7320) * prepend3 + (now_yy == 10286) * prepend4).permute(1,0)
        if task_eval == "snli":
            prepend1 = torch.tensor([[50256, 13830, 5118] for _ in range(n_inputs)]).permute(1,0)
            prepend2 = torch.tensor([[50256, 46458, 15806] for _ in range(n_inputs)]).permute(1,0)
            prepend3 = torch.tensor([[992, 20103, 19237] for _ in range(n_inputs)]).permute(1,0)
            now_yy = Y[0][range(n_inputs), len_cqs-1]
            prepend = ((now_yy == 39793) * prepend1 + (now_yy == 8500) * prepend2 + (now_yy == 25741) * prepend3).permute(1,0)

        now_cqs = torch.cat([prepend, now_cqs], dim=1)
        len_cqs = len_cqs + args.uat_len

        count_examples += n_inputs
        if not isinstance(args.test_batch_size, list):
            if count_examples % 300 == 0: # can handle bsz = 1,2,3,4,5,6; alter 300 for proper verbosing
                logger.info("done {}, total {}, now {}%".format(count_examples, n_examples, int(count_examples*10000.0/n_examples)/100))
        else:
            logger.info("done {}, total {}, now {}%".format(count_examples, n_examples, int(count_examples*10000.0/n_examples)/100))
        s_tokens_control = torch.zeros_like(s_tokens).requires_grad_() # this is the robust prefix psi to be optimized during inference
        optimizer = torch.optim.Adam([s_tokens_control], PMP_lr)

        for ii in range(PMP_iter):
            with torch.enable_grad():
                past = (s_tokens + s_tokens_control).unsqueeze(0).expand(n_inputs, -1, -1).cuda()
                bsz, seqlen, _ = past.shape
                past = past.view(bsz, seqlen, MODEL_CONFIG.n_layer * 2,
                                 MODEL_CONFIG.n_head, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head)#.type(torch.half)
                past = past.permute([2, 0, 3, 1, 4]).split(2)
                all_outputs = model(input_ids=now_cqs.cuda(), past=past)
                del past
                torch.cuda.empty_cache()
        
                outputs = all_outputs[0]
                if "gpt2" in args.model_name:
                    pasts = all_outputs[1]
                    all_hidden_states = all_outputs[2]

                loss_layer = torch.tensor(0.)
                for jj in range(args.tune_layer):
                    for kk in range(args.control_len):
                        tmp = all_hidden_states[jj][range(n_inputs), len_cqs-1-kk, :]
                        loss_layer = loss_layer + criterion_layer(PCA_proj(tmp, projs[kk][jj], args.test_batch_size==1), tmp)

                if not isinstance(args.test_batch_size, list):
                    if count_examples % 300 == 0:
                        logger.info("now PMP iter: {}, layer loss: {}".format(ii, loss_layer.data))
                else:
                    logger.info("now PMP iter: {}, layer loss: {}".format(ii, loss_layer.data))

                if loss_layer.data < loss_history:
                    loss_history = loss_layer.data
                    s_tokens_control_best = s_tokens_control

            optimizer.zero_grad()
            loss_layer.backward()
            optimizer.step() 

        past = (s_tokens + s_tokens_control).unsqueeze(0).expand(n_inputs, -1, -1).cuda()
        bsz, seqlen, _ = past.shape
        past = past.view(bsz, seqlen, MODEL_CONFIG.n_layer * 2,
                         MODEL_CONFIG.n_head, MODEL_CONFIG.n_embd // MODEL_CONFIG.n_head)
        past = past.permute([2, 0, 3, 1, 4]).split(2)
        all_outputs = model(input_ids=now_cqs.cuda(), past=past)
        del past
        torch.cuda.empty_cache()

        outputs = all_outputs[0]
        if "gpt2" in args.model_name:
            pasts = all_outputs[1]

        next_logits = outputs[range(n_inputs), len_cqs-1, :] / args.temperature_qa
        next_tokens = logits_to_tokens(next_logits).cpu()

        for i in range(n_inputs):
            max_tot_lens[cnt] = max_a_len + test_qadata[cnt][1] + args.uat_len
            qa_results[cnt] = now_cqs[i][:len_cqs[i]]
            if next_tokens[i] != SPECIAL_TOKEN_IDS["eos_token"]:
                qa_results[cnt] = torch.cat((now_cqs[i][:len_cqs[i]], next_tokens[i]))
                if len(qa_results[cnt]) not in [max_tot_lens[cnt], args.max_len]:
                    need_process.update([[cnt, None]])
                    if "gpt2" in args.model_name:
                        for layer_id in range(MODEL_CONFIG.n_layer):
                            all_pasts[layer_id][cnt] = pasts[layer_id][:, i, ..., :len_cqs[i], :].type(torch.float)
            cnt += 1
       
        if len(need_process) > int(1 * args.memory_sizes[0] / now_cqs.shape[1]):  # dynamic threshold to avoid out of memory
            sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

        del now_cqs
        torch.cuda.empty_cache()

    sample_sequence(model, need_process, qa_results, all_pasts, max_tot_lens)

    for i in range(len(test_qadata)):
        _, len_cq, _, _, Y, _, _, _ = test_qadata[i]
        Y = list(filter(lambda x: x != -1, Y))[:-1]  # remove eos
        Y = ' '.join([str(y) for y in Y]).split(str(SPECIAL_TOKEN_IDS["pad_token"]))
        Y = [TOKENIZER.decode(list(map(int, y.split()))) for y in Y]
        qa_results[i] = [TOKENIZER.decode(qa_results[i].tolist()[len_cq+args.uat_len:]), Y]
    get_test_score(task_eval, qa_results, score_dict)

    model_dir = model.model_dir
    ep = model.ep
    results_path = os.path.join(model_dir,"qa_{}_p{}_ep{}_uat_cl{}_tune{}_pmplr{}_pmpiter{}_testbs{}.csv".format(task_eval,args.preseqlen,ep+1,args.control_len,args.tune_layer, args.PMP_lr,args.PMP_iter, args.test_batch_size if not isinstance(args.test_batch_size, list) else "Auto"))
    if not args.debug:
        with open(results_path, "w",encoding="utf-8") as f:
            qa_writer = csv.writer(f,delimiter=',')
            qa_writer.writerow(["y","pred"])
            for pred, y in qa_results:
                qa_writer.writerow([y,pred])

    return model, score_dict

def get_test_score(task_eval,qa_results,score_dict):

    score = compute_metrics(
            qa_results,
            task=task_eval
    )
    score_dict[task_eval] = score


def test_one_to_many(task_load):
    score_dicts = []
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
    model.transformer.output_hidden_states = True
    model.transformer.output_attentions = False

    s_tokens = torch.load(s_tokens_path).cpu().to("cuda")
    proj_path = os.path.join(model_dir, 'Proj_{}_p{}_ep{}_cl{}.pth.tar'.format(task_load, args.preseqlen, ep+1, args.control_len))
    projs = torch.load(proj_path, map_location=torch.device('cuda'))

    model.ep = ep
    model.model_dir = model_dir
    logger.info("task: {}, epoch: {}".format(task_load, ep+1))
    score_dict = {k:None for k in args.tasks}
    with torch.no_grad():
        for task_eval in args.tasks:
            test_one_to_one(task_load, task_eval, model, score_dict, s_tokens, projs, PMP_lr=args.PMP_lr, PMP_iter=args.PMP_iter)
    logger.info("score: {}".format(score_dict))
    score_dicts.append(score_dict)
    del s_tokens
    torch.cuda.empty_cache()

    with open(os.path.join(model_dir, "p"+str(args.preseqlen)+'lr'+str(args.learning_rate)+"metricsep"+str(args.test_ep)+"_uat"+"_cl"+str(args.control_len)+"_tune"+str(args.tune_layer)+"_pmplr"+str(args.PMP_lr)+"_pmpiter"+str(args.PMP_iter)+"_testbs{}".format(args.test_batch_size if not isinstance(args.test_batch_size, list) else "Auto")+".json"),"w") as f:
        json.dump(score_dicts, f)

if __name__ == '__main__':
    if args.n_gpus > 1:
        raise NotImplementedError("test can be run with only one gpu currently!")
    
    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    init_logging(os.path.join(args.model_dir_root, 'log_test_p{}_lr{}_ep{}_uat_cl{}_tune{}_pmplr{}_pmpiter{}_testbs{}.txt'.format(args.preseqlen, args.learning_rate, args.test_ep, args.control_len, args.tune_layer, args.PMP_lr, args.PMP_iter, args.test_batch_size if not isinstance(args.test_batch_size, list) else "Auto")))
    logger.info('args = {}'.format(args))

    for task_load in args.tasks:
        test_one_to_many(task_load)

