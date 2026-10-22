import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='lastfm', help='Tmall/retailrocket/lastfm')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))

if dataset == "Tmall":
    num = 40728
elif opt.dataset == 'of':
    num = 2421
elif opt.dataset == 'clothing':
    num = 23034
elif opt.dataset=='sports':
    num =18358
elif opt.dataset=='baby':
    num =7051
else:
    raise Exception('Unknown Dataset!')

relation = []
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

#提取商品之间的共线关系
for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

#统计共现频次 adj1:[{1: 3, 2: 1, 3: 1}, {2: 1, 3: 1}, {3: 1}, {1: 1}]
for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

weight = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

#adj:[[1, 2], [2, 3], [3], [1]]邻居节点0邻居1和2
#weight:[[3, 1], [1, 1], [1], [1]]邻居权重（0，1）出现3次，（0，2）出现1次
pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))
