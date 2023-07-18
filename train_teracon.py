import torch
import torch.nn as nn
import data_loader as data_loader
import teracon as generator_recsys

import math
import numpy as np
import argparse

import random
import time
from tqdm import tqdm
from tqdm import trange
import collections

from utils import *
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--gpu_num', type=int, default=2,
                        help='Device (GPU) number')
    parser.add_argument('--epochs',type=int, default=100,
                        help='Total number of epochs')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--alpha', default=0.7, type=float,
                        help='Controls the contribution of the knowledge retention')
    parser.add_argument('--savepath',type=str, default='./saved_models/task2',
                        help='Save path of current model')
    parser.add_argument('--paths', type=str, default='./saved_models/task1.t7',
                        help='Load path of past model')
    parser.add_argument('--seed', type=int, default = 10,
                        help='Seed')
    parser.add_argument('--lr', type = float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--datapath', type=str, default='./ColdRec/original_desen_finetune_like_nouserID.csv',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/index.csv',
                        help='item index dictionary path')
    parser.add_argument('--split_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--smax',type=int, default = 50)
    parser.add_argument('--clipgrad',type=int, default = 1000)

    parser.add_argument('--batch', type=int,default=1024)
    parser.add_argument('--model_name', type=str,default='NextitNet')
    parser.add_argument('--n_tasks', type=int, default = 2,
                        help='The total number of tasks')
    args = parser.parse_args()
    
    
    """Set seed"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    """
    Load data
    "all_samples": consists of a sequence of item indices for each user and targets (labels).
    """
    dl = data_loader.Data_Loader_Sup({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index})
    items = dl.item_dict
    print("len(source)",len(items)) # The number of items.
    targets_ = dl.target_dict
    targets_len_ = len(targets_) #The number of targets.
    print('len(target)', targets_len_)
    all_samples = dl.example
    print('len(all_samples)',len(all_samples))

    # Shuffle the data.
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    #Split the data into train, validation, and test datasets.
    dev_sample_index = -1 * int(args.split_percentage * float(len(all_samples)))
    dev_sample_index_valid = int(dev_sample_index*0.75)
    train_set, valid_set_, test_set_ = all_samples[:dev_sample_index], all_samples[dev_sample_index:dev_sample_index_valid], all_samples[dev_sample_index_valid:]

    # Set GPU
    if args.gpu_num == 'cpu':
        args.device = 'cpu'
    else:
        args.device = torch.device("cuda:" + str(args.gpu_num) if torch.cuda.is_available() else "cpu")

    """
    Load information of previous tasks.
    "target_size": The list of the total number of past tasks' target.
    "task_dict": Model parameters of previous task.
    "current_task_dict": Current model parameters (before train current model). 
    """
    target_size, task_dict, current_task_dict = task_model(args)
    target_size.append(targets_len_) # Append the number of current task's target to list.


    """
    Set model parameters
    "item_size": the total number of unique items.
    "dilated_channels": Dimension of item embedding and hidden state.
    "target_size": The list of the total number of past tasks' target.
    "dilations": dilation of convolutional layers.
    "kernel_size": kernel size of convolutional layers.
    "learning_rate": model learning rate.
    "batch_size": training batch size.
    "task_embs": The upper bound and lower bound of the initialization distribution for the task embedding.
    "num_task": The total number of tasks.
    """
    model_para = {
        'item_size': len(items),
        'dilated_channels': 256,
        'target_item_size': target_size,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':args.lr,
        'batch_size':args.batch,
        'task_embs':[0,2],
        'num_task':args.n_tasks,
    }

    # Generate current model
    model = generator_recsys.NextItNet_Decoder(model_para).to(args.device)

    # Set the total number of tasks to generate previous model
    model_para['num_task'] = args.n_tasks - 1
    
    # Generate past tasks model
    past_models =  generator_recsys.NextItNet_Decoder(model_para).to(args.device)
    past_models.load_state_dict(task_dict,strict=False) # Load model parameters of previous task.
    for n,p in past_models.named_parameters():
            p.requires_grad = False
    
    # Load the model parameters of the previous task into the current model.
    model.load_state_dict(current_task_dict, strict=False)
    
    #Set loss function (criterion, criterion_)
    criterion = nn.CrossEntropyLoss() # Classification Loss
    criterion_ = nn.MSELoss() # Knowledge retention loss

    #Set optimizer (Adam)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_para['learning_rate'], weight_decay=0)
    
    # Set early stop
    count = 0
    best_acc = 0

    # Initialize the sampling ratio of Equation 14, 15 of paper.
    random_batch_size = [0.9 for _ in range(model_para['num_task'])]

    #Train
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        transfer_loss_t=0
        correct = 0
        total = 0
        batch_size = model_para['batch_size']
        batch_num = train_set.shape[0] / batch_size
        start = time.time()
        INFO_LOG("-------------------------------------------------------train")
        for batch_idx, batch_sam in enumerate(getBatch(train_set, batch_size)):
            """
            Split the inputs and targets (labels).
            The target value is located at the last index of each data line.
            """
            inputs = torch.LongTensor(batch_sam[:, :-1]).to(args.device)
            target = torch.LongTensor(batch_sam[:,-1]).to(args.device)

            # Sample the user for knowledge retention. Refer to Equation 14 of the paper.
            random_buffer = [[i for i in range(len(inputs))] for n in range(model_para['num_task'])]
            for buf in random_buffer:
                random.shuffle(buf)
            rand_idx = [np.array(random_buffer[i][:int(len(random_buffer[i])*random_batch_size[i])]) for i in range(len(random_buffer))]

            #Annealing trick. Refer to Equation 19 of paper.
            smax = args.smax
            r = batch_num
            s = (smax-1/smax)*batch_idx/r+1/smax
            optimizer.zero_grad()


            """
            Generate pseudo labels for knowlege retention for each task.
            Refer to Equation 9, 10, 16 of paper.
            The pseudo labels are stored in "pev_outputs" list for Equation 16.
            
            The task-specific masks are stored in "prev_masks" list for Equation 15.
            
            Inputs of model
            x: input data (User behavior sequence).
            s: value of s (annealing of s_max).
            masks_list: a list to store the task-specific masks for each layer.
            task_num: task index.
            gpu_num: device (gpu) number.
            onecall (default False): if True, return the last representation of sequence.
                                     if False, return all representation of sequence.
            new_task: Index of the newly emerged task.
            """
            prev_outputs, prev_masks = [], []
            for i in range(model_para['num_task']):
                if i ==0:
                    prev_o,prev_m = past_models(inputs[rand_idx[i],:-1],s,[], i,args.gpu_num,onecall=True,new_task=(args.n_tasks-1))
                else:
                    prev_o,prev_m = past_models(inputs[rand_idx[i],:],s,[], i,args.gpu_num,onecall=True,new_task=(args.n_tasks-1))
                prev_outputs.append(prev_o)
                prev_masks.append(prev_m)


            # Get outputs of newly emerged task using current model for Equation 18 in paper.
            # Newly emerged task masks are stored in "current_masks" list for Equation 15 in paper.
            # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
            outputs,current_masks = model(inputs,s,[], args.n_tasks-1, args.gpu_num,onecall=True,new_task=(args.n_tasks-1))


            # Get output of previous tasks (i.e., pseudo label predictions) using current model.
            # The pseudo label predictions are stored in "student_outpus" list for Equation 16 in paper.
            student_outputs, _masks = [], []
            for i in range(model_para['num_task']):
                if i ==0:
            # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
                    student_o,m = model(inputs[rand_idx[i],:-1],s,[], i, args.gpu_num,onecall=True,backward=True,new_task=(args.n_tasks-1))
                else:
            # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
                    student_o,m = model(inputs[rand_idx[i],:],s,[], i, args.gpu_num,onecall=True,backward=True,new_task=(args.n_tasks-1))
                student_outputs.append(student_o)
            
            #Calculate Knowledge Retention Loss (Equation 16 of the paper).
            transfer_loss = 0
            for i in range(len(student_outputs)):
                transfer_loss += random_batch_size[i]/sum(random_batch_size)*criterion_(student_outputs[i],prev_outputs[i])

            # Calculate classification loss of current task (Equation 18 of the paper).
            loss = criterion(outputs, target)

            # Calculate sampling ratio for each tasks. (Equation 15 of the paper).
            random_batch_size = [sample_ratio(prev_masks[i],current_masks) for i in range(len(student_outputs))]

            clipgrad = args.clipgrad

            if not torch.isfinite(loss):
                print("Occured Nan", loss)
                loss = 0
                total += 0
            else:
                loss += (args.alpha * transfer_loss)
                loss.backward()
                train_loss += loss.item()
                transfer_loss_t +=args.alpha * transfer_loss.item()
                
                # Use gradient clipping if needed
                thres_cosh = 50
                # for n,p in model.named_parameters():
                #     if ('.fec' in n):
                #         num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                #         den=torch.cosh(p.data)+1
                #         if p.grad != None:
                #             p.grad.data*=smax/s*num/den
                # torch.nn.utils.clip_grad_norm_(model.parameters(),clipgrad)
                optimizer.step()

            thres_emb = 6
            for n,p in model.named_parameters():
                if ('.ec' in n):
                    p.data=torch.clamp(p.data,-thres_emb,thres_emb)
                    
            _,predicted = outputs.max(1)
            total +=target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # We display the training results for each epoch, printing them out 100 times.
            if batch_idx % max(10, batch_num//100) == 0:
                INFO_LOG("epoch: {}\t {}/{}".format(epoch, batch_idx, batch_num))
                print('Transfer Loss: %.3f'%(transfer_loss_t/(batch_idx+1)))
                print('Loss: %.3f | Acc(hit@1): %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        end = time.time()

        model.eval()
        correct = 0
        total = 0
        batch_size_test = model_para['batch_size'] #Batch size of validation and test
        batch_num_test = valid_set_.shape[0] / batch_size
        list_ = [[] for i in range(6)]
        INFO_LOG("-------------------------------------------------------valid")
        with torch.no_grad():
            start = time.time()
            for batch_idx, batch_sam in enumerate(getBatch(valid_set_, batch_size_test)):
                inputs = torch.LongTensor(batch_sam[:, :-1]).to(args.device)
                target = torch.LongTensor(batch_sam[:,-1]).to(args.device)

                # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
                outputs, masks1 = model(inputs,smax,[], args.n_tasks-1,args.gpu_num,onecall=True,new_task=(args.n_tasks-1))

                list_toy = [[] for i in range(6)]
                output_mean = outputs
                
                if targets_len_ >(args.top_k+15): # If the number of targets are large, calculate metrics such as MRR, Hit, NDCG else calcuate accuracy only.
                    _, sort_idx_20 = torch.topk(output_mean, k=args.top_k + 15, sorted=True)
                    _, sort_idx_5 = torch.topk(output_mean, k=args.top_k, sorted=True)
                    result_ = accuracy_test(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), target.data.cpu().numpy(),
                            batch_idx, batch_num_test, epoch, args, list_toy)
                    for i in range(len(list_)):
                        list_[i] +=result_[i]
                _, predicted = output_mean.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            end = time.time()
            print('Acc(hit@1): %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
        if targets_len_ >(args.top_k+15): # If the number of targets are large, calculate metrics such as MRR, Hit, NDCG else calcuate accuracy only.
            INFO_LOG("Accuracy mrr_5: {}".format(sum(list_[0]) / float(len(list_[0]))))
            INFO_LOG("Accuracy mrr_20: {}".format(sum(list_[3]) / float(len(list_[3]))))
            INFO_LOG("Accuracy hit_5: {}".format(sum(list_[1]) / float(len(list_[1]))))
            INFO_LOG("Accuracy hit_20: {}".format(sum(list_[4]) / float(len(list_[4]))))
            INFO_LOG("Accuracy ndcg_5: {}".format(sum(list_[2]) / float(len(list_[2]))))
            INFO_LOG("Accuracy ndcg_20: {}".format(sum(list_[5]) / float(len(list_[5]))))
                
        # Conduct test at best validation (store the model parameters)
        # The last recorded test score is considered as the test score at best validation score, as the test is performed whenever a new validation score is updated.
        if epoch == 0:
            best_acc = (100.*correct/total)
            count = 0
            print('-----testing in best validation-----')
            list__ = [[] for i in range(6)]
            # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
            if targets_len_ >(args.top_k+15): # If the number of targets are large, calculate metrics such as MRR, Hit, NDCG else calcuate accuracy only.
                model_test(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = args.n_tasks-1,new_task=args.n_tasks-1)
            else:
                model_test_acc(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = args.n_tasks-1, new_task=args.n_tasks-1)

        else:
            if best_acc < (100.*correct/total):
                best_acc = (100.*correct/total)
                count = 0
                print('-----testing in best validation-----')
                list__ = [[] for i in range(6)]
                # We substract 1 at args.n_tasks (i.e., new_task = args.n_tasks-1) Because, the index of tasks start from 0.
                if targets_len_ >(args.top_k+15): # If the number of targets are large, calculate metrics such as MRR, Hit, NDCG else calcuate accuracy only.
                    model_test(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = args.n_tasks-1,new_task=args.n_tasks-1)
                else:
                    model_test_acc(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = args.n_tasks-1, new_task=args.n_tasks-1)
            else: # Early stop
                count+=1

        if count == 5: # Early stop
            break
        print('count', count)
        INFO_LOG("TIME FOR EPOCH During Training: {}".format(end - start))
        INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))


if __name__ == '__main__':
    curr_preds_5 = []
    rec_preds_5 = []
    ndcg_preds_5 = []
    curr_preds_20 = []
    rec_preds_20 = []
    ndcg_preds_20 = []
    main()

