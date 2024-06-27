
def return_meta(dataset):
    if dataset == "Movielens":
        meta_paths = {#["um", "mu"]["um", "mg", "gm", "mu"]
            "user": [["um", "mu"], ["um", "mg", "gm", "mu"]],
            "movie": [["mu", "um"], ["mg", "gm"]],#,
        }
        user_key = "user"
        item_key = "movie"
    elif dataset == "Amazon":
        meta_paths = {
            "user": [["ui", "ic", "ci", "iu"]],
            # , ["ui", "ib", "bi", "iu"], ["ui", "ic", "ci", "iu"],["ui", "iu"], ["iv", "vi"]],, ["ic", "ci"], ["ib", "bi"]
            "item": [["iu", "ui"], ["ic", "ci"], ["ib", "bi"]]
        }
        user_key = "user"
        item_key = "item"
    elif dataset == "Yelp":
        meta_paths = {
            # "user": [["ub", "bu"],["uc", "cu"]],
            # , ["ub", "bu", "ub", "bu"], ["bc", "cb"], ["bi", "ib"]
            "user": [["ub", "bu"],["uc", "cu"]],
            "business": [["bu", "ub"],["bc", "cb"], ["bi", "ib"]],
        }
        user_key = "user"
        item_key = "business"
    elif dataset == "Dbbook":
        meta_paths = {
            "user": [["ub", "bu"],],
            "book": [["bu", "ub"], ["ba", "ab"]],
        }
        user_key = "user"
        item_key = "book"
    elif dataset == "LastFM":
        meta_paths = {
            "user": [["ua", "au"], ],
            "artist": [["au", "ua"], ["at", "ta"]],
        }
        user_key = "user"
        item_key = "artist"
    elif dataset == "wy_Yelp":
        meta_paths = {
            # "user": [["ub", "bu"],["uc", "cu"]],
            # , ["ub", "bu", "ub", "bu"], ["bc", "cb"], ["bi", "ib"]
            "user": [["ub", "bu"]],
            "business": [["bu", "ub"], ["bc", "cb"], ["bi", "ib"]],
        }
        user_key = "user"
        item_key = "business"
    else:
        print("Available datasets: Movielens, amazon, yelp.")
        raise NotImplementedError
    return meta_paths,user_key,item_key