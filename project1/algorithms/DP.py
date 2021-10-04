import json
import math
import numpy as np
from tqdm import tqdm
import os.path as osp
import time
import numpy as np
from pprint import pprint


class DP:
    def __init__(self, cfg=None, loader=None):
        self.time_start = time.time()
        self.loader = loader
        self.cfg = cfg
        self.mode = self.cfg.QUERY[0]
        self.min_cost = math.inf
        self.pos = []
        self.dps = []
        self.init_dp()
        self.res = {"cost": math.inf}
        self.delta = 2  # gap

    def init_dp(self):
        if self.mode == 2:
            for l in self.loader.seq_len[1:]:
                self.dps.append(
                    np.zeros((len(self.loader.query[1]) + 1, l + 1), dtype=np.int64))
            for dp in self.dps:
                for i in range(dp.shape[0]):
                    dp[i][0] = 2*i
                for i in range(dp.shape[1]):
                    dp[0][i] = 2*i
        elif self.mode == 3:
            for i in range(1, len(self.loader.seq_len)):
                for j in range(i + 1, len(self.loader.seq_len)):
                    self.dps.append(np.zeros(
                        (self.loader.seq_len[0] + 1, self.loader.seq_len[i] + 1, self.loader.seq_len[j] + 1), dtype=np.float16))
                    self.pos.append([i-1, j-1])
            for dp in self.dps:
                for i in range(dp.shape[0]):
                    dp[i][0][0] = 4*i
                for i in range(dp.shape[1]):
                    dp[0][i][0] = 4*i
                for i in range(dp.shape[2]):
                    dp[0][0][i] = 4*i

    def alpha(self, c1, c2):
        return 3 if not c1 == c2 else 0

    def search_2_seq(self):
        for k in tqdm(range(len(self.dps))):
            dp = self.dps[k]
            # store route
            query = [['' for j in range(dp.shape[1])]
                     for i in range(dp.shape[0])]
            value = [['' for j in range(dp.shape[1])]
                     for i in range(dp.shape[0])]
            for i in range(1, dp.shape[0]):
                query[i][0] = self.loader.query[1][0:i]
                value[i][0] = '-'*i
            for i in range(1, dp.shape[1]):
                query[0][i] = '-'*i
                value[0][i] = self.loader.database[k][0:i]
            # compute dp
            for i in range(1, dp.shape[0]):
                for j in range(1, dp.shape[1]):
                    c1 = self.loader.query[1][i-1]
                    c2 = self.loader.database[k][j-1]
                    cand = [dp[i-1][j-1] + self.alpha(c1, c2),
                            dp[i-1][j] + self.delta,
                            dp[i][j-1] + self.delta,
                            ]
                    dp[i][j] = min(cand)
                    if cand[0] == dp[i][j]:
                        query[i][j] = query[i-1][j-1]+c1
                        value[i][j] = value[i-1][j-1]+c2
                    elif cand[1] == dp[i][j]:
                        query[i][j] = query[i-1][j]+c1
                        value[i][j] = value[i-1][j]+'-'
                    else:
                        query[i][j] = query[i][j-1]+'-'
                        value[i][j] = value[i][j-1]+c2
            cur_cost = dp[dp.shape[0]-1][dp.shape[1]-1]
            self.res = {"mode": 'pairwise' if self.mode == 2 else '3-seq',
                        "cost": cur_cost, "query": query[dp.shape[0]-1][dp.shape[1]-1],
                        "value": value[dp.shape[0]-1][dp.shape[1]-1],
                        } if cur_cost < self.res['cost'] else self.res

    def search_3_seq(self):
        for l in tqdm(range(len(self.dps))):
            dp = self.dps[l]
            # store route
            query = [[['' for k in range(dp.shape[2])]
                      for j in range(dp.shape[1])]
                     for i in range(dp.shape[0])]
            value1 = [[['' for k in range(dp.shape[2])]
                       for j in range(dp.shape[1])]
                      for i in range(dp.shape[0])]
            value2 = [[['' for k in range(dp.shape[2])]
                       for j in range(dp.shape[1])]
                      for i in range(dp.shape[0])]

            for i in range(1, dp.shape[0]):
                query[i][0][0] = self.loader.query[1][0:i]
                value1[i][0][0] = '-'*i
                value2[i][0][0] = '-'*i
            for i in range(1, dp.shape[1]):
                query[0][i][0] = '-'*i
                value1[0][i][0] = self.loader.database[self.pos[l][0]][0:i]
                value2[0][i][0] = '-'*i
            for i in range(1, dp.shape[2]):
                query[0][0][i] = '-'*i
                value1[0][0][i] = '-'*i
                value2[0][0][i] = self.loader.database[self.pos[l][1]][0:i]

            # Compute 3 surfaces
            for i in range(1, dp.shape[0]):
                for j in range(1, dp.shape[1]):
                    c1 = self.loader.query[1][i-1]
                    c2 = self.loader.database[self.pos[l][0]][j-1]
                    cand = [
                        dp[i-1][j-1][0] +
                        self.alpha(c1, c2) + 2 * self.delta,

                        dp[i-1][j][0] + 2 * self.delta,
                        dp[i][j-1][0] + 2 * self.delta,
                    ]
                    dp[i][j][0] = min(cand)
                    if cand[0] == dp[i][j][0]:
                        query[i][j][0] = query[i-1][j-1][0]+c1
                        value1[i][j][0] = value1[i-1][j-1][0]+c2
                        value2[i][j][0] = value2[i-1][j-1][0]+'-'
                    elif cand[1] == dp[i][j][0]:
                        query[i][j][0] = query[i-1][j][0]+c1
                        value1[i][j][0] = value1[i-1][j][0]+'-'
                        value2[i][j][0] = value2[i-1][j][0]+'-'
                    elif cand[2] == dp[i][j][0]:
                        query[i][j][0] = query[i][j-1][0]+'-'
                        value1[i][j][0] = value1[i][j-1][0]+c2
                        value2[i][j][0] = value2[i][j-1][0]+'-'
            for i in range(1, dp.shape[1]):
                for j in range(1, dp.shape[2]):
                    c1 = self.loader.database[self.pos[l][0]][i-1]
                    c2 = self.loader.database[self.pos[l][1]][j-1]
                    cand = [
                        dp[0][i-1][j-1] +
                        self.alpha(c1, c2) + 2 * self.delta,

                        dp[0][i-1][j] + 2 * self.delta,
                        dp[0][i][j-1] + 2 * self.delta,
                    ]
                    dp[0][i][j] = min(cand)
                    if cand[0] == dp[0][i][j]:
                        query[0][i][j] = query[0][i-1][j-1]+'-'
                        value1[0][i][j] = value1[0][i-1][j-1]+c1
                        value2[0][i][j] = value2[0][i-1][j-1]+c2
                    elif cand[1] == dp[0][i][j]:
                        query[0][i][j] = query[0][i-1][j]+'-'
                        value1[0][i][j] = value1[0][i-1][j]+c1
                        value2[0][i][j] = value2[0][i-1][j]+'-'
                    elif cand[2] == dp[0][i][j]:
                        query[0][i][j] = query[0][i][j-1]+'-'
                        value1[0][i][j] = value1[0][i][j-1]+'-'
                        value2[0][i][j] = value2[0][i][j-1]+c2
            for i in range(1, dp.shape[0]):
                for j in range(1, dp.shape[2]):
                    c1 = self.loader.query[1][i-1]
                    c2 = self.loader.database[self.pos[l][1]][j-1]
                    cand = [
                        dp[i-1][0][j-1] +
                        self.alpha(c1, c2) + 2 * self.delta,

                        dp[i-1][0][j] + 2 * self.delta,
                        dp[i][0][j-1] + 2 * self.delta,
                    ]
                    dp[i][0][j] = min(cand)
                    if cand[0] == dp[i][0][j]:
                        query[i][0][j] = query[i-1][0][j-1]+c1
                        value1[i][0][j] = value1[i-1][0][j-1]+'-'
                        value2[i][0][j] = value2[i-1][0][j-1]+c2
                    elif cand[1] == dp[i][0][j]:
                        query[i][0][j] = query[i-1][0][j]+c1
                        value1[i][0][j] = value1[i-1][0][j]+'-'
                        value2[i][0][j] = value2[i-1][0][j]+'-'
                    elif cand[2] == dp[i][0][j]:
                        query[i][0][j] = query[i][0][j-1]+'-'
                        value1[i][0][j] = value1[i][0][j-1]+'-'
                        value2[i][0][j] = value2[i][0][j-1]+c2

            # compute in cubic
            for i in range(1, dp.shape[0]):  # query
                for j in range(1, dp.shape[1]):  # seq1
                    for k in range(1, dp.shape[2]):  # seq2
                        c1 = self.loader.query[1][i-1]
                        c2 = self.loader.database[self.pos[l][0]][j-1]
                        c3 = self.loader.database[self.pos[l][1]][k-1]
                        cand = [
                            dp[i-1][j-1][k-1] +
                            self.alpha(c1, c2) + self.alpha(c2,
                                                            c3) + self.alpha(c1, c3),

                            dp[i-1][j-1][k] +
                            self.alpha(c1, c2) + 2 * self.delta,
                            dp[i-1][j][k-1] +
                            self.alpha(c1, c3) + 2 * self.delta,
                            dp[i][j-1][k-1] +
                            self.alpha(c2, c3) + 2 * self.delta,

                            dp[i][j][k-1] + 2 * self.delta,
                            dp[i][j-1][k] + 2 * self.delta,
                            dp[i-1][j][k] + 2 * self.delta
                        ]
                        dp[i][j][k] = min(cand)
                        if cand[0] == dp[i][j][k]:
                            query[i][j][k] = query[i-1][j-1][k-1]+c1
                            value1[i][j][k] = value1[i-1][j-1][k-1]+c2
                            value2[i][j][k] = value2[i-1][j-1][k-1]+c3
                        elif cand[1] == dp[i][j][k]:
                            query[i][j][k] = query[i-1][j-1][k]+c1
                            value1[i][j][k] = value1[i-1][j-1][k]+c2
                            value2[i][j][k] = value2[i-1][j-1][k]+'-'
                        elif cand[2] == dp[i][j][k]:
                            query[i][j][k] = query[i-1][j][k-1]+c1
                            value1[i][j][k] = value1[i-1][j][k-1]+'-'
                            value2[i][j][k] = value2[i-1][j][k-1]+c3
                        elif cand[3] == dp[i][j][k]:
                            query[i][j][k] = query[i][j-1][k-1]+'-'
                            value1[i][j][k] = value1[i][j-1][k-1]+c2
                            value2[i][j][k] = value2[i][j-1][k-1]+c3
                        elif cand[4] == dp[i][j][k]:
                            query[i][j][k] = query[i][j][k-1]+'-'
                            value1[i][j][k] = value1[i][j][k-1]+'-'
                            value2[i][j][k] = value2[i][j][k-1]+c3
                        elif cand[5] == dp[i][j][k]:
                            query[i][j][k] = query[i][j-1][k]+'-'
                            value1[i][j][k] = value1[i][j-1][k]+c2
                            value2[i][j][k] = value2[i][j-1][k]+'-'
                        elif cand[6] == dp[i][j][k]:
                            query[i][j][k] = query[i-1][j][k]+c1
                            value1[i][j][k] = value1[i-1][j][k]+'-'
                            value2[i][j][k] = value2[i-1][j][k]+'-'
            cur_cost = dp[dp.shape[0]-1][dp.shape[1]-1][dp.shape[2]-1]
            self.res = {"mode": '3-seq',
                        "cost": cur_cost, "query": query[dp.shape[0]-1][dp.shape[1]-1][dp.shape[2]-1],
                        "value1": value1[dp.shape[0]-1][dp.shape[1]-1][dp.shape[2]-1],
                        "value2": value2[dp.shape[0]-1][dp.shape[1]-1][dp.shape[2]-1],
                        } if cur_cost < self.res['cost'] else self.res
            # pprint(self.res)

    def save_res(self):
        self.time_end = time.time()
        self.res["run_time"] = "{:.4f} sec".format(
            self.time_end-self.time_start)
        pprint(self.res)
        self.res = {k: str(v) for k, v in self.res.items()}
        json.dump(self.res, open(osp.join(self.cfg.RESULT.DIR,
                                          'DP.json'), 'w'))

    def search(self):
        print("Searching...")
        if self.mode == 2:
            self.search_2_seq()
        elif self.mode == 3:
            self.search_3_seq()
        self.save_res()
