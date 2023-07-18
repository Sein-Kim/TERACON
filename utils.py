import numpy as np
import time
import math
import torch
import copy
import torch.nn as nn

"""
Load information of previous tasks.
"target_size": The list of the total number of past tasks' target.
"model_dict": Model parameters of previous task.
"current_task_dict": Current model parameters (before train current model). 
"""
def task_model(args):
    # Load information of previous tasks.
    model_path = args.paths
    new_dict = torch.load(model_path, map_location=torch.device(args.device))
    model_dict = new_dict['net']
    
    # For newly emergy task (i.e., for current model), we expand and reinitialize the mlp at Equation 12 in paper.
    current_model_dict = copy.deepcopy(model_dict)
    for key in list(current_model_dict.keys()):
        if 'mlp' in key:
            del current_model_dict[key]
    
    # Get the list of the total number of past tasks' target.
    target_size = []
    for d in range(args.n_tasks-1):
        target_size.append(model_dict['task_classifier.{}.weight'.format(d)].shape[0])
    return [target_size, model_dict, current_model_dict]


def inference_model(args):
    # Load information of previous tasks.
    model_path = args.paths
    new_dict = torch.load(model_path, map_location=torch.device(args.device))
    model_dict = new_dict['net']
        
    # Get the list of the total number of past tasks' target.
    target_size = []
    for d in range(args.n_tasks):
        target_size.append(model_dict['task_classifier.{}.weight'.format(d)].shape[0])
    return [target_size, model_dict]


# Get Batch
def getBatch(data, batch_size):
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffle_indices]

    start_inx = 0
    end_inx = batch_size

    while end_inx < len(data):
        batch = data[start_inx:end_inx]
        start_inx = end_inx
        end_inx += batch_size
        yield batch


def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))


"""Computes the accuracy over the k top predictions for the specified values of k"""
def accuracy_test(pred_items_5, pred_items_20, target, batch_idx, batch_num, epoch,args, list_): # output: [batch_size, 20] target: [batch_size]
    curr_preds_5_ = list_[0]
    rec_preds_5_ = list_[1]
    ndcg_preds_5_ = list_[2]
    curr_preds_20_ = list_[3]
    rec_preds_20_ = list_[4]
    ndcg_preds_20_ = list_[5]
    for bi in range(pred_items_5.shape[0]):

        true_item=target[bi]
        predictmap_5={ch : i for i, ch in enumerate(pred_items_5[bi])}
        predictmap_20 = {ch: i for i, ch in enumerate(pred_items_20[bi])}

        rank_5 = predictmap_5.get(true_item)
        rank_20 = predictmap_20.get(true_item)
        if rank_5 == None:
            curr_preds_5_.append(0.0)
            rec_preds_5_.append(0.0)
            ndcg_preds_5_.append(0.0)
        else:
            MRR_5 = 1.0/(rank_5+1)
            Rec_5 = 1.0#3
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
            curr_preds_5_.append(MRR_5)
            rec_preds_5_.append(Rec_5)#4
            ndcg_preds_5_.append(ndcg_5)  # 4
        if rank_20 == None:
            curr_preds_20_.append(0.0)
            rec_preds_20_.append(0.0)#2
            ndcg_preds_20_.append(0.0)#2
        else:
            MRR_20 = 1.0/(rank_20+1)
            Rec_20 = 1.0#3
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
            curr_preds_20_.append(MRR_20)
            rec_preds_20_.append(Rec_20) # 4
            ndcg_preds_20_.append(ndcg_20)  # 4
    
    return [curr_preds_5_,rec_preds_5_,ndcg_preds_5_,curr_preds_20_,rec_preds_20_,ndcg_preds_20_]


"""
Test the performance of model with continually learning.
"model_test" fucntion calculate metrics such as MRR, Hit, NDCG.
Save the results of those metrics in args.save_path.txt.
"task_num": task_num represents the index of the task for which we want to perform inference (testing).
"new_task": new_task corresponds to the index of the newly emerged task (i.e., total number of tasks - 1).
"""
def model_test(model_para, test_set, model, epoch, args, list_,smax,backward,task_num,new_task):
    best_acc = 0
    model.eval()
    correct = 0
    total = 0
    batch_size = model_para['batch_size']

    batch_num = test_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(test_set, batch_size)):
            inputs = torch.LongTensor(batch_sam[:, :-1]).to(args.device)
            target = torch.LongTensor(batch_sam[:,-1]).to(args.device)
            student_out,masks = model(inputs,smax,[], task_num, args.gpu_num,onecall=True,backward=backward,new_task=new_task)
            output_mean = student_out

            list_toy = [[] for i in range(6)]

            _, sort_idx_20 = torch.topk(output_mean, k=args.top_k + 15, sorted=True)  # [batch_size, 20]
            _, sort_idx_5 = torch.topk(output_mean, k=args.top_k, sorted=True)  # [batch_size, 5]
            list__ = accuracy_test(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), target.data.cpu().numpy(),
                     batch_idx, batch_num, epoch, args, list_toy)
            for i in range(len(list_)):
                list_[i] +=list__[i]

            _, predicted = output_mean.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        end = time.time()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': model.state_dict(),
            'acc(hit@1)': acc
        }
        torch.save(state, '%s.t7' % (args.savepath))

    print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

    INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
        epoch, args.epochs, batch_num))
    INFO_LOG("Accuracy mrr_5: {}".format(sum(list_[0]) / float(len(list_[0]))))
    INFO_LOG("Accuracy mrr_20: {}".format(sum(list_[3]) / float(len(list_[3]))))
    INFO_LOG("Accuracy hit_5: {}".format(sum(list_[1]) / float(len(list_[1]))))
    INFO_LOG("Accuracy hit_20: {}".format(sum(list_[4]) / float(len(list_[4]))))
    INFO_LOG("Accuracy ndcg_5: {}".format(sum(list_[2]) / float(len(list_[2]))))
    INFO_LOG("Accuracy ndcg_20: {}".format(sum(list_[5]) / float(len(list_[5]))))

    f = open('%s.txt' %(args.savepath), 'a')
    word = "epoch: {}\t total_epoch:{}\t total_batches:{}\n".format(
        epoch, args.epochs, batch_num)
    f.write(word)
    f.write("Accuracy mrr_5: {}\n".format(sum(list_[0]) / float(len(list_[0]))))
    f.write("Accuracy mrr_20: {}\n".format(sum(list_[3]) / float(len(list_[3]))))
    f.write("Accuracy hit_5: {}\n".format(sum(list_[1]) / float(len(list_[1]))))
    f.write("Accuracy hit_20: {}\n".format(sum(list_[4]) / float(len(list_[4]))))
    f.write("Accuracy ndcg_5: {}\n".format(sum(list_[2]) / float(len(list_[2]))))
    f.write("Accuracy ndcg_20: {}\n".format(sum(list_[5]) / float(len(list_[5]))))
    f.close()




"""
Test the performance of model (after Task 1).
"model_test_acc" fucntion calculate accuracy.
Save the results of accuracy in args.save_path.txt.
"task_num": task_num represents the index of the task for which we want to perform inference (testing).
"new_task": new_task corresponds to the index of the newly emerged task (i.e., total number of tasks - 1).
"""
def model_test_acc(model_para, test_set, model, epoch, args, list_,smax,backward,task_num,new_task):
    best_acc = 0
    model.eval()
    correct = 0
    total = 0
    iftest=True
    batch_size = model_para['batch_size']
    print(task_num)
    batch_num = test_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(test_set, batch_size)):
            inputs = torch.LongTensor(batch_sam[:, :-1]).to(args.device)
            target = torch.LongTensor(batch_sam[:,-1]).to(args.device)
            masks_list = []

            student_out,masks = model(inputs,smax,masks_list, task_num, args.gpu_num,onecall=True,backward=backward,new_task=new_task)
            output_mean = student_out

            _, predicted = output_mean.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        end = time.time()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': model.state_dict(),
            'acc(hit@1)': acc
        }
        torch.save(state, '%s.t7' % (args.savepath))
    print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

    INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
        epoch, args.epochs, batch_num))
    f = open('%s.txt' %(args.savepath), 'a')
    word = "epoch: {}\t total_epoch:{}\t total_batches:{}\n".format(
        epoch, args.epochs, batch_num)
    f.write(word)
    f.write('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))
    f.close()


def sample_ratio(mask1, mask2):
    gate = nn.Sigmoid()
    cos = nn.CosineSimilarity(dim=1,eps=1e-6)
    jaccard = 0
    len_jaccard = 0
    for i in range(len(mask1)):
        jaccard +=gate(6*cos(mask1[i],mask2[i])).detach().cpu().item()
        len_jaccard+=1
    random_batch_size = 1 - jaccard/len_jaccard
    return random_batch_size


"""
Test the performance of model without continually learning.
In the case of Task 1, the model is evaluated using the "model_test_" function.
However, after training Task 2, the model performance for Task 1 be evaluated using the "model_test" function instead of "model_test_".
Because the model conduct continual learning.

"model_test_" fucntion calculate metrics such as MRR, Hit, NDCG.
Save the results of those metrics in args.save_path.txt.
"""
def model_test_(model_para, test_set, model, epoch, args, list_,smax):#This is function for only Task 1.
    best_acc = 0
    model.eval()
    correct = 0
    total = 0
    batch_size = 512
    batch_num = test_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(test_set, batch_size)):
            inputs, target = torch.LongTensor(batch_sam[:,:-1]).to(args.device), torch.LongTensor(batch_sam[:,-1]).to(args.device).view([-1])

            masks_list = []
            outputs, _ = model(inputs,smax,masks_list,0, args.gpu_num,onecall=True)
            output_mean = outputs


            list_toy = [[] for i in range(6)]
            _, sort_idx_20 = torch.topk(output_mean, k=args.top_k + 15, sorted=True)  # [batch_size, 20]
            _, sort_idx_5 = torch.topk(output_mean, k=args.top_k, sorted=True)  # [batch_size, 5]
            list__ = accuracy_test(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), target.data.cpu().numpy(),
                     batch_idx, batch_num, epoch, args, list_toy)
            for i in range(len(list_)):
                list_[i] +=list__[i]

            _, predicted = output_mean.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        end = time.time()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': model.state_dict(),
            'acc(hit@1)': acc
        }
        torch.save(state, '%s.t7' % (args.savepath))

    print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

    INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
        epoch, args.epochs, batch_num))
    INFO_LOG("Accuracy mrr_5: {}".format(sum(list_[0]) / float(len(list_[0]))))
    INFO_LOG("Accuracy mrr_20: {}".format(sum(list_[3]) / float(len(list_[3]))))
    INFO_LOG("Accuracy hit_5: {}".format(sum(list_[1]) / float(len(list_[1]))))
    INFO_LOG("Accuracy hit_20: {}".format(sum(list_[4]) / float(len(list_[4]))))
    INFO_LOG("Accuracy ndcg_5: {}".format(sum(list_[2]) / float(len(list_[2]))))
    INFO_LOG("Accuracy ndcg_20: {}".format(sum(list_[5]) / float(len(list_[5]))))

    f = open('%s.txt' %(args.savepath), 'a')
    word = "epoch: {}\t total_epoch:{}\t total_batches:{}\n".format(
        epoch, args.epochs, batch_num)
    f.write(word)
    f.write("Accuracy mrr_5: {}\n".format(sum(list_[0]) / float(len(list_[0]))))
    f.write("Accuracy mrr_20: {}\n".format(sum(list_[3]) / float(len(list_[3]))))
    f.write("Accuracy hit_5: {}\n".format(sum(list_[1]) / float(len(list_[1]))))
    f.write("Accuracy hit_20: {}\n".format(sum(list_[4]) / float(len(list_[4]))))
    f.write("Accuracy ndcg_5: {}\n".format(sum(list_[2]) / float(len(list_[2]))))
    f.write("Accuracy ndcg_20: {}\n".format(sum(list_[5]) / float(len(list_[5]))))
    f.close()