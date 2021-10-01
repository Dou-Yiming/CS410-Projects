import os.path as osp


class Data_loader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.database = []
        self.query = cfg.QUERY
        self.load_database()
        self.seq_len = [len(self.query[1])] + \
            [len(s) for s in self.database]
        self.max_seq_len = max(self.seq_len)

    def load_database(self):
        with open(osp.join(self.cfg.DATASET.DATA_DIR,
                           'MSA_database.txt')) as f:
        # with open(osp.join(self.cfg.DATASET.DATA_DIR,
        #                    'toy_database.txt')) as f:
            self.database = f.readlines()
            self.database = [line.strip("\n") for line in self.database]
