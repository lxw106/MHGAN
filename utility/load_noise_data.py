import torch
import scipy
import numpy as np
import scipy.sparse as sp
def load_noise_data(path, Train_edges, u_num, i_num):

    """
        以下用 gowalla 的 train 数据集举例：
            inter_users: 将 userID 冗余存储, [0, 0, ... 1, 1, ... 2, 2, ...], 810128
            inter_items: 将 itemID 冗余存储, [0, 1, 2, ...], 810128
            unique_users: 将 userID 唯一存储, [0, 1, 2, ...], 29858
            inter_num: 记录 item 总数, 810128
            pos_length: 分别记录每个 userID 交互的物品总数量, [127, 49, ...], 29858

            user_id: 循环变量，记录当前 userID
            pos_id: 循环变量，记录当前 userID 交互的各个 itemID
    """
    inter_users, inter_items, unique_users = [], [], []
    inter_num = 0
    pos_length = []
    with open(path, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            temp = line.strip()
            arr = [int(i) for i in temp.split(" ")]
            user_id, pos_id = arr[0], arr[1:]
            unique_users.append(user_id)

            # if len(pos_id) < 1:
            #     print(user_id, pos_id)
            #     break
            # self.num_users = max(self.num_users, user_id)
            # self.num_items = max(self.num_items, max(pos_id))

            inter_users.extend([user_id] * len(pos_id))
            pos_length.append(len(pos_id))
            inter_items.extend(pos_id)
            inter_num += len(pos_id)
            line = f.readline()
    # a = inter_users
    # b = inter_items
    row = torch.Tensor(inter_users).long()
    col = torch.Tensor(inter_items).long()
    data = torch.ones_like(row)
    noise_ui_net = scipy.sparse.coo_matrix((data, (row, col)), shape=(u_num, i_num))#.to(device)
    # noise_ui_net = torch.sparse_coo_tensor(indices=torch.stack([row, col]), values=torch.ones_like(row).int(), size=(u_num, i_num)).to(device)
    Cha_num_edges = len(row) - Train_edges
    return noise_ui_net, Cha_num_edges
#coo
def read_ratings(file_name):
        """
            以下用 gowalla 的 train 数据集举例：
                inter_users: 将 userID 冗余存储, [0, 0, ... 1, 1, ... 2, 2, ...], 810128
                inter_items: 将 itemID 冗余存储, [0, 1, 2, ...], 810128
                unique_users: 将 userID 唯一存储, [0, 1, 2, ...], 29858
                inter_num: 记录 item 总数, 810128
                pos_length: 分别记录每个 userID 交互的物品总数量, [127, 49, ...], 29858

                user_id: 循环变量，记录当前 userID
                pos_id: 循环变量，记录当前 userID 交互的各个 itemID
        """
        inter_users, inter_items, unique_users = [], [], []
        inter_num = 0
        pos_length = []
        with open(file_name, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()
                arr = [int(i) for i in temp.split(" ")]
                user_id, pos_id = arr[0], arr[1:]
                unique_users.append(user_id)

                # if len(pos_id) < 1:
                #     print(user_id, pos_id)
                #     break
                # self.num_users = max(self.num_users, user_id)
                # self.num_items = max(self.num_items, max(pos_id))

                inter_users.extend([user_id] * len(pos_id))
                pos_length.append(len(pos_id))
                inter_items.extend(pos_id)
                inter_num += len(pos_id)
                line = f.readline()

        return np.array(unique_users), np.array(inter_users), np.array(inter_items), inter_num, pos_length

