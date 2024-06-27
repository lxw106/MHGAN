import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MCL")

    parser.add_argument("--seed", type=int, default=2023, help="random seed for init")
    parser.add_argument("--model", default="TAHIN", help="Model Name")
    parser.add_argument(
        "--dataset",
        default="Movielens",
        help="Dataset to use, Movielens, Amazon, Yelp, Dbbook, LastFM",
    )
    parser.add_argument("--multicore", type=int, default=0, help="use multiprocessing or not in test")
    parser.add_argument("--data_path", nargs="?", default="./data", help="Input data path.")
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout", type=bool, default=False, help="consider node dropout or not")
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]', help='Output sizes of every layer')
    parser.add_argument('--test_batch_size', type=int, default=100, help='batch size')
    parser.add_argument("--mess_keep_prob", nargs='?', default='[0.1, 0.1, 0.1]', help="ratio of node dropout")
    parser.add_argument("--node_keep_prob", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--dim', type=int, default=128, help='embedding size')

    parser.add_argument('--drop_ratio', type=float, default=0.3, help='l2 regularization weight')

    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')

    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")#0.005  0.001 0.001

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of processes to construct batches",
    )

    parser.add_argument(
        "--in_size",
        default=128,
        type=int,
        help="Initial dimension size for entities.",
    )
    parser.add_argument(
        "--out_size",
        default=128,
        type=int,
        help="Output dimension size for entities.",
    )

    parser.add_argument(
        "--num_heads", default=1, type=int, help="Number of attention heads"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default="0",
        help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout.")#0.3 0.5 0.3

    parser.add_argument('--topK', nargs='?', default='[20]', help='size of Top-K')

    parser.add_argument("--verbose", type=int, default=10, help="Test interval")

    parser.add_argument('--regs', nargs='?', default='[1e-4]',
                        help='Regularizations.')

    parser.add_argument('--GCNLayer', type=int, default=3, help="the layer number of GCN")
    parser.add_argument('--LODA', type=int, default=1, help="Whether to load model parameters( yes:1, no:0 ). After you have trained once, you can set it to 1 to save time")
    parser.add_argument('--n_perturbation', type=int, default=5, help="Number of perturbations (Modify one edge per disturbance)")
    parser.add_argument('--noise_name', default="0.2", help="noise file name")# "0.05" "0.1"
    parser.add_argument('--lambda0', type=float, default=0.001, help='weight for L0 loss on laplacian matrix.')
    parser.add_argument('--gamma', type=float, default=-0.50)
    parser.add_argument('--zeta', type=float, default=1.05)
    parser.add_argument('--temperature_decay', type=float, default=0.98)
    parser.add_argument('--init_temperature', type=float, default=2.0)
    parser.add_argument('--ssl_beta', type=float, default=0.001, help='weight of loss with ssl')
    return parser.parse_args()
args = parse_args()
