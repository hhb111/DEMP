import time
import argparse
import pickle
from model import *
from utils import *
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = None


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='Tmall/retailrocket/lastfm')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=10, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-4, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2] 传播几层
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', default=True,action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--interests', type=int, default=3, help='The number of interests')  # [1, 2, 3, 4, 5]
parser.add_argument('--id_interests', type=int, default=None,
                    help='Number of ID interests. If omitted, keep the legacy model default.')
parser.add_argument('--image_interests', type=int, default=None,
                    help='Number of image interests. If omitted, keep the legacy model default.')
parser.add_argument('--text_interests', type=int, default=None,
                    help='Number of text interests. If omitted, keep the legacy model default.')
parser.add_argument('--beta', type=float, default=0.01, help='Beta for the interests regularization')   # [0, 0.001, 0.005, 0.01, 0.02, 0.05]
parser.add_argument('--length', type=float, default=8, help='eta in the paper')          # [8, 10, 12, 14, 16, 18]
parser.add_argument('--gamma', type=float, default=0.4,
                    help='Counterfactual target weight for the visual modality; target is [0, gamma, 1-gamma].')
parser.add_argument('--lambda1', type=float, default=None,
                    help='Weight of the contrastive alignment loss. If omitted, use the legacy value 0.5.')
parser.add_argument('--lambda2', type=float, default=None,
                    help='Weight of the diversity loss. If omitted, use the legacy dataset-specific beta behavior.')
parser.add_argument('--lambda3', type=float, default=None,
                    help='Weight of the counterfactual constraint. If omitted, use the legacy value 1.0.')
parser.add_argument('--seed', type=int, default=2025, help='Random seed.')
parser.add_argument('--variant', type=str, default='cf_detach_backbone',
                    help='Experiment label. The current code path uses the counterfactual branch detached from the backbone.')
parser.add_argument('--gate_variant', type=str, default='seq_proto',
                    choices=['seq_proto', 'seq_no_proto', 'static', 'item'],
                    help='Modality fusion gate used for gate ablations.')
parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                    help='Directory containing dataset folders, or a specific dataset folder.')
parser.add_argument('--quick', action='store_true',
                    help='Run a small deterministic subset for quick smoke tests.')
parser.add_argument('--train_limit', type=int, default=None,
                    help='Maximum number of training instances used in quick mode.')
parser.add_argument('--valid_limit', type=int, default=None,
                    help='Maximum number of validation instances used in quick mode.')
parser.add_argument('--test_limit', type=int, default=None,
                    help='Maximum number of test instances used in quick mode.')

opt = parser.parse_args()


def limit_dataset(data, limit, seed):
    if limit is None:
        return data
    inputs, targets = data
    n_samples = len(inputs)
    if limit <= 0 or limit >= n_samples:
        return data
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_samples, size=limit, replace=False)
    limited_inputs = [inputs[i] for i in indices]
    limited_targets = [targets[i] for i in indices]
    return limited_inputs, limited_targets


def resolve_dataset_dir(data_dir, dataset):
    candidates = []

    if data_dir:
        data_dir = os.path.abspath(data_dir)
        candidates.append(data_dir)
        candidates.append(os.path.join(data_dir, dataset))

    candidates.extend([
        os.path.join(BASE_DIR, 'datasets', dataset),
        os.path.join(os.getcwd(), 'datasets', dataset),
        os.path.abspath(os.path.join(BASE_DIR, '..', 'datasets', dataset)),
        os.path.join(os.getcwd(), dataset),
    ])

    seen = set()
    checked = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(candidate)
        if os.path.exists(os.path.join(candidate, 'train.txt')) and os.path.exists(os.path.join(candidate, 'test.txt')):
            return candidate

    raise FileNotFoundError(
        "Cannot find dataset '%s'. "
        "Pass --data_dir as the parent directory containing dataset folders." % dataset
    )


def main():
    init_seed(opt.seed)

    if opt.quick:
        opt.epoch = min(opt.epoch, 1)
        if opt.train_limit is None:
            opt.train_limit = 2048
        if opt.valid_limit is None:
            opt.valid_limit = 512
        if opt.test_limit is None:
            opt.test_limit = 512

    if opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_global = 1
        opt.dropout_gcn = 1
        opt.dropout_local = 0.7
        opt.beta = 0.02
        opt.interests = 5
        opt.length = 8
    elif opt.dataset == 'of':
        num_node = 2421
        opt.n_iter = 1
        opt.dropout_global = 0.2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
        opt.beta = 0.02
        opt.interests = 1
        opt.length = 6
    elif opt.dataset == 'clothing':
        num_node = 23034
        opt.n_iter = 1
        opt.dropout_global = 0.1
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0
        opt.length = 10
        opt.beta = 0.01
        opt.interests = 5
    elif opt.dataset=='sports':
        num_node = 18358
        opt.n_iter = 1
        opt.dropout_global = 0.3
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.2
        opt.length =8
        opt.beta = 0
        opt.interests = 2
    elif opt.dataset=='baby':
        num_node = 7051
        opt.n_iter = 1
        opt.dropout_global = 0.2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.2
        opt.length =4
        opt.beta = 0.01
        opt.interests = 1
    else:
        raise Exception('Unknown Dataset!')

    dataset_dir = resolve_dataset_dir(opt.data_dir, opt.dataset)
    opt.dataset_dir = dataset_dir
    train_data = pickle.load(open(os.path.join(dataset_dir, 'train.txt'), 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = pickle.load(open(os.path.join(dataset_dir, 'test.txt'), 'rb'))
    else:
        test_data = pickle.load(open(os.path.join(dataset_dir, 'test.txt'), 'rb'))
        valid_data = test_data

    if opt.quick:
        train_data = limit_dataset(train_data, opt.train_limit, opt.seed)
        valid_data = limit_dataset(valid_data, opt.valid_limit, opt.seed + 1)
        test_data = limit_dataset(test_data, opt.test_limit, opt.seed + 2)

    adj = pickle.load(open(os.path.join(dataset_dir, 'adj_' + str(opt.n_sample_all) + '.pkl'), 'rb'))
    num = pickle.load(open(os.path.join(dataset_dir, 'num_' + str(opt.n_sample_all) + '.pkl'), 'rb'))
    #adj:[[1, 2], [2, 3], [3], [1]]邻居节点0邻居1和2
    #num:[[3, 1], [1, 1], [1], [1]]邻居权重（0，1）出现3次，（0，2）出现1次
    # print(len(adj), len(num))
    train_data = Data(train_data)
    # print(train_data.__dict__)
    test_data = Data(test_data)
    valid_data = Data(valid_data) 

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(Hybrid_MoE(opt, num_node, adj, num))

    if opt.quick:
        print('[Quick mode] train=%d valid=%d test=%d seed=%d variant=%s gate=%s' % (
            len(train_data), len(valid_data), len(test_data), opt.seed, opt.variant, opt.gate_variant))
    start = time.time()
    best_result = [0, 0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        r10, mrr10, n10, r20, mrr20, n20, cov = train(model, train_data, valid_data)
        print("Generating visualization...")
        #visualize_three_modalities(model, num_samples=200)
        cov = cov * 100 / (num_node - 1)
        current_result = [r10, mrr10, n10, r20, mrr20, n20, cov]
        flag = 0
        for i, value in enumerate(current_result):
            if value >= best_result[i]:
                best_result[i] = value
                best_epoch[i] = epoch
                flag = 1
        print('Current Result:')
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tNDCG@10:\t%.4f\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tNDCG@20:\t%.4f\tCov@20:\t%.4f' %
              (r10, mrr10, n10, r20, mrr20, n20, cov))
        print('Best Result:')
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tNDCG@10:\t%.4f\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tNDCG@20:\t%.4f\tCov@20:\t%.4f\t\tEpoch:\t%d,%d,%d,%d,%d,%d,%d' % (
            best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5], best_result[6],
            best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5], best_epoch[6]))
        r10, mrr10, n10, r20, mrr20, n20, cov = test(model,test_data)
        # 1. 跑数据
        #sim_scores = analyze_macro_disentanglement(model, test_data)
        #plot_similarity_histogram(sim_scores, dataset_name="Amazon_Sports")
        print('test Result:')
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tNDCG@10:\t%.4f\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tNDCG@20:\t%.4f\tCov@20:\t%.4f' %
              (r10, mrr10, n10, r20, mrr20, n20, cov * 100 / (num_node - 1)))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

if __name__ == '__main__':
    main()
