import pickle as pkl
from utility.dataloader import Data
from utility.return_meta import return_meta
import time
import torch
import torch.optim as optim
import os
import utility.parser
import utility.batch_test
import dgl
from MHGAN import MHGAN
from utility.load_noise_data import load_noise_data
import warnings
warnings.filterwarnings('ignore')


def main():
    args = utility.parser.parse_args()
    utility.batch_test.set_seed(2024)
    # step 1: Check device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # step 2: Load data
    g_file = open(os.path.join(args.data_path + args.dataset, "hg.pkl"), "rb")
    g = pkl.load(g_file)
    g_file.close()
    g = g.to(device)
    dataset = Data(args.data_path + args.dataset)
    meta_paths,user_key,item_key = return_meta(args.dataset)
    print("Data loaded.")

    # step 3: Create pure model and training components
    model = MHGAN(args.dataset, g, meta_paths, user_key, item_key, args.in_size, args.out_size, args.num_heads, args.dropout, device)
    model = model.to(device)
    print("Pure Model created.")
    weight_file = './result/' + args.dataset + "/ROHE_HAN_emb" + ".pth.tar"
    print(f"load and save to {weight_file}")

    #Choose between testing or training(Model needs to be loaded before testing)
    if args.LODA:
        try:
            model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            print(f"loaded model weights from {weight_file}")
        except:
            # print("Please train the " + args.dataset + " dataset using the ROHE model first (making LODA = 0)")
            raise NotImplementedError("Please train the " + args.dataset + " dataset using the ROHE_HAN model first (making LODA = 0)")
    else:
        # Training
        print("Start training.")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_report_recall = 0.
        best_report_ndcg = 0.
        best_report_epoch = 0
        early_stop = 0
        for epoch in range(args.epochs):
            since = time.time()
            # Training and validation using a full graph
            if 1:
                result = utility.batch_test.Test(dataset, model, device, eval(args.topK), args.multicore,
                                                 args.test_batch_size, long_tail=False)
                if (result['recall'][0] > best_report_recall) and (result['ndcg'][0]>(best_report_ndcg-0.002)):
                    torch.save(model.state_dict(), weight_file)
                    print("save...")
                    early_stop = 0
                    best_report_epoch = epoch + 1
                    best_report_recall = result['recall'][0]
                    best_report_ndcg = result['ndcg'][0]
                else:
                    early_stop += 1

                if early_stop >= 20:
                    print("early stop! best epoch:", best_report_epoch, "bset_recall:", best_report_recall, ',best_ndcg:',
                          best_report_ndcg)
                    with open('./result/' + args.dataset + "/ROHE_result.txt", "a") as f:
                        f.write(str(args.dataset) + ", ")
                        f.write(str(args.lr) + ", ")
                        f.write(str(args.dropout) + ", ")
                        f.write(str(best_report_epoch) + " ")
                        f.write(str(best_report_recall) + " ")
                        f.write(str(best_report_ndcg) + "\n")
                    break
                else:
                    print("recall:", result['recall'], ",precision:", result['precision'], ',ndcg:', result['ndcg'])

            model.train()
            sample_data = dataset.sample_data_to_train_all()
            users = torch.Tensor(sample_data[:, 0]).long()
            pos_items = torch.Tensor(sample_data[:, 1]).long()
            neg_items = torch.Tensor(sample_data[:, 2]).long()

            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            users, pos_items, neg_items = utility.batch_test.shuffle(users, pos_items, neg_items)
            num_batch = len(users) // args.batch_size + 1
            average_loss = 0.
            average_reg_loss = 0.

            for batch_i, (batch_users, batch_positive, batch_negative) in enumerate(
                    utility.batch_test.mini_batch(users, pos_items, neg_items, batch_size=args.batch_size)):
                batch_mf_loss, batch_emb_loss = model.bpr_loss(batch_users, batch_positive, batch_negative)
                batch_emb_loss = eval(args.regs)[0] * batch_emb_loss
                batch_loss = batch_emb_loss + batch_mf_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                average_loss += batch_mf_loss.item()
                average_reg_loss += batch_emb_loss.item()

            average_loss = average_loss / num_batch
            average_reg_loss = average_reg_loss / num_batch
            time_elapsed = time.time() - since
            print("\t Epoch: %4d| train time: %.3f | train_loss:%.4f + %.4f" % (
                epoch + 1, time_elapsed, average_loss, average_reg_loss))
    if args.LODA:
        Train_edges = dataset.num_train
        adj_ui, Cha_num_edges  = load_noise_data(os.path.join(args.data_path + args.dataset, "random_noise_train_0.3.txt"), Train_edges, dataset.num_users, dataset.num_items)
        adj_ui= adj_ui.tocsc()
        edges_dict_path = os.path.join(args.data_path + args.dataset, "edges_dict.pkl")
        with open(edges_dict_path, 'rb') as f:
            edges_dict = pkl.load(f)
        Train_edges = dataset.num_train
        if args.dataset=="Movielens":
            edges_dict[("user", "um", "movie")] = adj_ui.nonzero()
            edges_dict[("movie", "mu", "user")] = adj_ui.transpose().nonzero()
        elif args.dataset=="Amazon":
            edges_dict[("user", "ui", "item")] = adj_ui.nonzero()
            edges_dict[("item", "iu", "user")] = adj_ui.transpose().nonzero()
        elif args.dataset == "Yelp":
            edges_dict[("user", "ub", "business")] = adj_ui.nonzero()
            edges_dict[("business", "bu", "user")] = adj_ui.transpose().nonzero()
        elif args.dataset == "LastFM":
            edges_dict[("user", "ua", "artist")] = adj_ui.nonzero()
            edges_dict[("artist", "au", "user")] = adj_ui.transpose().nonzero()
        elif args.dataset == "Dbbook":
            edges_dict[("user", "ub", "book")] = adj_ui.nonzero()
            edges_dict[("book", "bu", "user")] = adj_ui.transpose().nonzero()
        else:
            raise NotImplementedError(f"Haven't supported {args.dataset} yet!")
        attack_g = dgl.heterograph(edges_dict).to(device)

        # pure test or attack test
        print("\nTest informations", "\n(1)Model: ROHE_HAN", "\n(2)Dataset: ", args.dataset, "\n(3)Attack rate K: ", args.n_perturbation,
              "\n(4)Train edges: ", Train_edges)
        #, "\n(5)Change the number of edges: ", Cha_num_edges, "\n(6)Change edge rate: {:.2f}%".format((Cha_num_edges/Train_edges)*100)

        print("\nPure Test...")
        result = utility.batch_test.Test(dataset, model, device, eval(args.topK), args.multicore, args.test_batch_size, long_tail=False)
        print("recall:", result['recall'], ",precision:", result['precision'], ',ndcg:', result['ndcg'])
        pure_recall =  result['recall'][0]
        pure_ndcg =  result['ndcg'][0]

        print("\nAttack Test...")
        attack_model = MHGAN(args.dataset, attack_g, meta_paths, user_key, item_key, args.in_size, args.out_size, args.num_heads, args.dropout, device)
        attack_model = attack_model.to(device)
        attack_model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        result = utility.batch_test.Test(dataset, attack_model, device, eval(args.topK), args.multicore, args.test_batch_size, long_tail=False)
        print("recall:", result['recall'], ",precision:", result['precision'], ',ndcg:', result['ndcg'])
        attack_recall = result['recall'][0]
        attack_ndcg = result['ndcg'][0]

        print("\nrecall fall: {:.2f}%, ndcg fall: {:.2f}%".format((1-attack_recall/pure_recall)*100, (1-attack_ndcg/pure_ndcg)*100))






if __name__ == '__main__':
    main()