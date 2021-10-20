import json
import math
from tqdm import tqdm
import os.path as osp
import time
import heapq
from pprint import pprint
import numpy as np


def M_dist(pos1, pos2):
    # Manhattan distance
    ans = 0
    for i in range(len(pos1)):
        ans += abs(pos1[i]-pos2[i])
    return ans


def H_dist(pos1, pos2):
    ans = []
    for i in range(len(pos1)):
        ans.append(abs(pos1[i]-pos2[i]))
    return (max(ans)-min(ans)) * 2


def E_dist(pos1, pos2):
    # Euclidean distance
    return math.sqrt(
        sum([math.pow(pos1[i]-pos2[i], 2) for i in range(len(pos1))])
    )


class Node:
    def __init__(self, pos, g, f, query=None, seq1=None, seq2=None):
        self.pos = pos
        self.g = g
        self.f = f
        self.query = query
        self.seq1 = seq1
        self.seq2 = seq2

    def __lt__(self, other):
        return self.f < other.f


class Astar:
    def __init__(self, cfg=None, loader=None):
        self.time_start = time.time()
        self.loader = loader
        self.cfg = cfg
        self.mode = self.cfg.QUERY[0]
        self.goal = None
        self.delta = 2
        self.res = {"cost": math.inf}

    def alpha(self, c1, c2):
        return 3 if not c1 == c2 else 0

    def init_lists_2(self):
        self.open_list = []
        heapq.heappush(self.open_list, Node(
            [0, 0], 0, H_dist([0, 0], self.goal), query='', seq1=''))
        self.closed_list = []

    def init_lists_3(self):
        self.open_list = []
        heapq.heappush(self.open_list, Node([0, 0, 0], 0, 0))
        self.closed_list = []

    def init_goal(self, goal):
        self.goal = goal

    def align_pair(self, seq1, seq2):
        """
        Align one pair of seqs
        """
        # re-initialization
        self.init_goal([len(seq1), len(seq2)])
        self.init_lists_2()
        # search
        while not len(self.open_list) == 0:
            head = heapq.heappop(self.open_list)
            if head.pos in self.closed_list:
                continue
            else:
                self.closed_list.append(head.pos)
            if head.pos == self.goal:  # reach the goal
                assert head.f == head.g
                return head.f, head.query, head.seq1
            p0 = head.pos[0]
            p1 = head.pos[1]
            neighbours = [[p0 + 1, p1],
                          [p0, p1 + 1],
                          [p0+1, p1+1]]
            for l in range(len(neighbours)):
                n = neighbours[l]
                if n in self.closed_list or\
                        n[0] > self.goal[0] or n[1] > self.goal[1]:
                    continue
                if l == 0:
                    cost_added = self.delta
                    q = head.query + seq1[p0]
                    s = head.seq1 + '-'
                elif l == 1:
                    cost_added = self.delta
                    q = head.query + '-'
                    s = head.seq1+seq2[p1]
                else:
                    cost_added = self.alpha(seq1[p0], seq2[p1])
                    q = head.query + seq1[p0]
                    s = head.seq1+seq2[p1]
                g = head.g + cost_added
                h = H_dist(n, self.goal)
                f = g + h
                heapq.heappush(self.open_list, Node(n, g, f, query=q, seq1=s))

    def align_pair_dp(self, seq1, seq2, dp):
        for i in range(1, dp.shape[0]):
            dp[i][0] = dp[i - 1][0] + self.delta
            for j in range(1, dp.shape[1]):
                c1 = seq1[i-1]
                c2 = seq2[j-1]
                cand = [dp[i-1][j-1] + self.alpha(c1, c2),
                        dp[i-1][j] + self.delta,
                        dp[i][j-1] + self.delta,
                        ]
                dp[i][j] = min(cand)
        return dp[dp.shape[0]-1][dp.shape[1]-1]

    def search_2_seq(self):
        """
        Search the optimal alignment of all pairs
        """
        print("Searching...")
        query = self.loader.query[1]
        for seq in tqdm(self.loader.database):
            pair_cost, q, s = self.align_pair(query, seq)
            if pair_cost < self.res['cost']:
                self.res['cost'] = pair_cost
                self.res['query'] = q
                self.res['value'] = s

    def init_dp(self, seq1, seq2):
        dp = np.zeros((len(seq1), len(seq2)))
        for i in range(dp.shape[1]):
            dp[0][i] = 2*i
        return dp

    def search_3_seq(self):
        query = self.loader.query[1]
        # Compute 3-seq alignment
        print("Searching...")
        for i in tqdm(range(len(self.loader.database))):
            for j in range(i + 1, len(self.loader.database)):
                seq1 = self.loader.database[i]
                seq2 = self.loader.database[j]
                self.init_goal([len(query), len(seq1), len(seq2)])
                self.init_lists_3()
                # search
                while not len(self.open_list) == 0:
                    head = heapq.heappop(self.open_list)
                    if head.pos in self.closed_list:
                        continue
                    else:
                        self.closed_list.append(head.pos)
                    if head.pos == self.goal:  # reach the goal
                        self.res['cost'] = head.g if head.g < self.res['cost'] else self.res['cost']
                        break
                    p0 = head.pos[0]
                    p1 = head.pos[1]
                    p2 = head.pos[2]
                    neighbours = [[p0 + 1, p1 + 1, p2 + 1],
                                  [p0 + 1, p1 + 1, p2],
                                  [p0 + 1, p1, p2 + 1],
                                  [p0, p1 + 1, p2 + 1],
                                  [p0 + 1, p1, p2],
                                  [p0, p1 + 1, p2],
                                  [p0, p1, p2 + 1]]
                    for l in range(len(neighbours)):
                        n = neighbours[l]
                        if n in self.closed_list or\
                                n[0] > self.goal[0] or\
                                n[1] > self.goal[1] or\
                                n[2] > self.goal[2]:
                            continue
                        if l == 0:
                            cost_added = self.alpha(
                                query[p0], seq1[p1]) + self.alpha(
                                    seq1[p1], seq2[p2]) + self.alpha(query[p0], seq2[p2])
                        elif l == 1:
                            cost_added = self.alpha(
                                query[p0], seq1[p1]) + 2*self.delta
                        elif l == 2:
                            cost_added = self.alpha(
                                query[p0], seq2[p2]) + 2*self.delta
                        elif l == 3:
                            cost_added = self.alpha(
                                seq1[p1], seq2[p2]) + 2*self.delta
                        else:
                            cost_added = 2*self.delta
                        g = head.g + cost_added
                        h = H_dist(n, self.goal)
                        f = g + h
                        heapq.heappush(self.open_list, Node(n, g, f))

    def save_res(self):
        self.time_end = time.time()
        self.res["run_time"] = "{:.4f} sec".format(
            self.time_end-self.time_start)
        pprint(self.res)
        self.res = {k: str(v) for k, v in self.res.items()}
        json.dump(self.res, open(osp.join(self.cfg.RESULT.DIR,
                                          'Astar.json'), 'w'))

    def search(self):
        if self.mode == 2:
            self.search_2_seq()
        elif self.mode == 3:
            self.search_3_seq()
        self.save_res()
