import os
import time
import torch
import pickle
import argparse

from models.tisasrec import TiSASRec, TiSASRecWithoutMetadata
from models.sasrec import SASRec
from tqdm import tqdm
from utils import *

USE_BASELINE_MODEL = False
ENRICH_WITH_METADATA = False

#* app_price, app_average_score, app_num_reviews, review_rating
METADATA_NUM = 4

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.00005, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--time_span', default=256, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, user_num, item_num, time_num] = dataset
num_batch = len(user_train) // args.batch_size
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

try:
    relation_matrix = pickle.load(open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'rb'))
except:
    relation_matrix = Relation(user_train, user_num, args.maxlen, args.time_span)
    pickle.dump(relation_matrix, open('data/relation_matrix_%s_%d_%d.pickle'%(args.dataset, args.maxlen, args.time_span),'wb'))

sampler = DataSampler(
    user_train,
    user_num,
    item_num,
    relation_matrix,
    batch_size=args.batch_size,
    maxlen=args.maxlen,
    n_workers=3,
)

model = SASRec if USE_BASELINE_MODEL else \
\
TiSASRec(
    item_num,
    args=args,
    metadata_num=METADATA_NUM,
    category_num=100,
).to(args.device) if ENRICH_WITH_METADATA else \
\
TiSASRecWithoutMetadata(item_num, args=args).to(args.device)

for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_uniform_(param.data)
    except:
        pass # just ignore those failed init layers

model.train() # enable model training

epoch_start_idx = 1
if args.state_dict_path is not None:
    try:
        model.load_state_dict(torch.load(args.state_dict_path))
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except:
        print('failed loading state_dicts, pls check file path: ', end="")
        print(args.state_dict_path)

if args.inference_only:
    model.eval()
    t_test = evaluate(model, dataset, args)
    print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

bce_criterion = torch.nn.BCEWithLogitsLoss()
# add weight decay for l2 regularization on embedding vectors during training
adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.l2_emb)

T = 0.0
t0 = time.time()

for epoch in range(epoch_start_idx, args.num_epochs + 1):
    if args.inference_only: break # skip training if in inference mode

    epoch_losses = []
    for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, time_seq, time_matrix, pos, neg = sampler.next_batch() # tuples to ndarray
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        time_seq, time_matrix = np.array(time_seq), np.array(time_matrix)

        pos_logits, neg_logits = model(seq, time_matrix, pos, neg, metadata=None)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

        # print("\nsanity check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0

        adam_optimizer.zero_grad()
        indices = np.where(pos != 0) # mask padding items

        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])

        # manual L2 regularization
        # use either weight_decay in optimizer or add to loss manually
        # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        # for param in model.abs_pos_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        # for param in model.abs_pos_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        # for param in model.time_matrix_K_emb.parameters(): loss += args.l2_emb * torch.norm(param)
        # for param in model.time_matrix_V_emb.parameters(): loss += args.l2_emb * torch.norm(param)

        loss.backward()
        adam_optimizer.step()
        epoch_losses.append(loss.detach().cpu().item())

        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

    #* run validation
    model.eval()
    t1 = time.time() - t0
    T += t1
    print('\nEvaluating', end='')
    t_valid = evaluate_valid(model, dataset, args)
    print('epoch:%d, time: %f(s), average training loss: %f, valid (NDCG@10: %.4f, HR@10: %.4f)'
        % (epoch, T, float(np.mean(epoch_losses)), t_valid[0], t_valid[1]))

    # f.write(str(t_valid) + ' ' + str(t_test) + '\n')
    # f.flush()
    t0 = time.time()
    model.train()

    # save model after each epoch
    print('\nsaving model...\n')
    folder = args.dataset + '_' + args.train_dir
    fname = 'TiSASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
    fname = fname.format(epoch, args.lr, args.num_blocks, args.num_heads, args.hidden_size, args.maxlen)
    torch.save(model.state_dict(), os.path.join(folder, fname))
    print('\nmodel saved!\n')


f.close()
sampler.close()
print("Done")