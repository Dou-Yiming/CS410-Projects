import random
from random import randint as rint
import json
import math
from tqdm import tqdm
import os.path as osp
import time
from pprint import pprint


class Gene:
    def __init__(self, traits):
        self.traits = traits
        self.gene = self.traits2gene(traits)
        self.length = len(self.gene)
        self.fittness = None
        self.res = {}

    def __lt__(self, other):
        return self.fittness < other.fittness

    def __gt__(self, other):
        return self.fittness > other.fittness

    def traits2gene(self, traits):
        gene = []
        for t in traits:
            gene += t
        return gene


class GA:
    def __init__(self, cfg=None, loader=None, cost_func=None):
        self.time_start = time.time()
        self.loader = loader
        self.cfg = cfg
        self.mode = self.cfg.QUERY[0]  # 2: pairwise; 3: 3-seq
        self.cost_func = cost_func
        self.res = None
        self.init_ppl()

    def add_bar(self, seq, traits):
        # add '-'s in the seq
        seq = list(seq)
        for i in range(len(traits)):
            seq.insert(traits[i] + i, '-')
        return ''.join(seq)

    def init_ppl(self):
        trait_ppl = []
        print('Population Initialization...')
        for p in tqdm(range(self.cfg.GA.PPL)):
            trait_ppl.append([sorted([rint(0, len(self.loader.query[1])-1)for i in range(self.loader.max_seq_len - len(self.loader.query[1]))])] +
                             [sorted([rint(0, len(self.loader.database[j])-1)for i in range(self.loader.max_seq_len - len(self.loader.database[j]))])
                              for j in range(len(self.loader.database))])
        self.gene_ppl = [Gene(traits) for traits in trait_ppl]

    def gene2traits(self, gene):
        traits = []
        pos = 0
        seq_len = self.loader.seq_len
        for n in seq_len:
            traits.append(sorted(gene[pos:pos+self.loader.max_seq_len-n]))
            pos += (self.loader.max_seq_len-n)
        return traits

    def eval_ppl(self):
        trait_ppl = [self.gene2traits(g.gene) for g in self.gene_ppl]
        if self.mode == 2:
            for i in range(len(self.gene_ppl)):  # for evry individual
                individual_cost = []
                # query
                q = self.add_bar(self.loader.query[1], trait_ppl[i][0])
                # database
                for j in range(len(self.loader.database)):  # for every seq
                    s = self.add_bar(
                        self.loader.database[j], trait_ppl[i][j + 1])
                    individual_cost.append(self.cost_func(q, s))
                self.gene_ppl[i].fittness = max([1000/(cost + 1e-8)  # avoid divide by 0
                                                 for cost in individual_cost])
        elif self.mode == 3:
            for i in range(len(self.gene_ppl)):  # for evry individual
                individual_cost = []
                # query
                q = self.add_bar(self.loader.query[1], trait_ppl[i][0])
                for j in range(len(self.loader.database)):
                    for k in range(j+1, len(self.loader.database)):
                        s1 = self.add_bar(
                            self.loader.database[j], trait_ppl[i][j + 1])
                        s2 = self.add_bar(
                            self.loader.database[k], trait_ppl[i][k + 1])
                        individual_cost.append(self.cost_func(
                            q, s1)+self.cost_func(q, s2)+self.cost_func(s1, s2))
                self.gene_ppl[i].fittness = max([1000/(cost + 1e-8)  # avoid divide by 0
                                                 for cost in individual_cost])

    def crossover(self, gene_pair):
        """
        Generate two offsprings of the pair of Gene
        pair: [gene1, gene2]
        mode: 1 or 2
        mode = 1 ===> single-point-crossover
        mode = 2 ===> double-point-crossover
        """
        length = gene_pair[0].length
        gene1, gene2 = gene_pair[0].gene, gene_pair[1].gene
        gene_new1, gene_new2 = [], []
        if self.cfg.GA.CROSSOVER.MODE == 2:
            pos1 = random.randrange(1, length)
            pos2 = random.randrange(1, length)
            for i in range(length):
                if min(pos1, pos2) <= i < max(pos1, pos2):
                    gene_new1.append(gene1[i])
                    gene_new2.append(gene2[i])
                else:
                    gene_new1.append(gene2[i])
                    gene_new2.append(gene1[i])
        # TODO: mode=1
        if random.random() < self.cfg.GA.MUTATION.PROB:
            gene_new1 = self.mutation(gene_new1)
            gene_new2 = self.mutation(gene_new2)
        data_new1, data_new2 = self.gene2traits(
            gene_new1), self.gene2traits(gene_new2)
        return [Gene(data_new1), Gene(data_new2)]

    def mutation(self, gene):
        """
        Randomly change one element in gene
        bound: the pair of lower and upper bound of mutation element
        """
        length = len(gene)
        pos = random.randrange(0, length)

        seq_no = 0
        tmp_pos = pos
        while tmp_pos >= self.loader.max_seq_len -\
                self.loader.seq_len[seq_no]:
            tmp_pos -= (self.loader.max_seq_len - \
                self.loader.seq_len[seq_no])
            seq_no += 1

        bound = [1, max(self.gene2traits(gene)[seq_no])]
        gene[pos] = random.randint(bound[0], bound[1])
        return gene

    def selection(self):
        """
        Randomly select k individuals from the population
        individuals with higher fittness are more likely to be chosen
        """
        s_popu = sorted(self.gene_ppl, reverse=True)
        sum_fittness = sum(p.fittness for p in self.gene_ppl)
        # print("Best fittness: {:.6f}".format(
        #     max(p.fittness for p in self.gene_ppl)))
        # print("Mean fittness: {:.6f}".format(sum_fittness/len(self.gene_ppl)))
        chosen = []  # The chosen individuals
        k = self.cfg.GA.NEXT_GEN
        for i in range(k):
            thres = random.random() * sum_fittness
            cur_sum_fittness = 0
            for individual in s_popu:
                cur_sum_fittness += individual.fittness
                if cur_sum_fittness >= thres:
                    chosen.append(individual)
                    break
        return sorted(chosen, reverse=False)

    def select_best(self):
        """
        Select the best individual
        """
        return max(self.gene_ppl)

    def select_worst(self):
        """
        Select the worst individual
        """
        return min(self.gene_ppl)

    def cost_traits(self, traits):
        """
        Computes the cost of traits
        """
        ans = {}
        ans['mode'] = 'pairwise' if self.mode == 2 else '3-seq'
        ans['cost'] = [math.inf]
        q = self.add_bar(self.loader.query[1], traits[0])
        if self.mode == 2:
            for j in range(len(self.loader.database)):  # for every seq
                s = self.add_bar(self.loader.database[j], traits[j + 1])
                cur_cost = self.cost_func(q, s)
                if cur_cost < ans['cost'][0]:
                    ans['cost'] = [cur_cost]
                    ans['query'] = [q]
                    ans['value'] = [s]
        elif self.mode == 3:
            for j in range(len(self.loader.database)):
                for k in range(j+1, len(self.loader.database)):
                    s1 = self.add_bar(
                        self.loader.database[j], traits[j + 1])
                    s2 = self.add_bar(
                        self.loader.database[k], traits[k + 1])
                    cur_cost = self.cost_func(
                        q, s1)+self.cost_func(q, s2)+self.cost_func(s1, s2)
                    if cur_cost < ans['cost'][0]:
                        ans['cost'] = [cur_cost]
                        ans['query'] = [q]
                        ans['value1'] = [s1]
                        ans['value2'] = [s2]
        return ans

    def add_res(self, cur_res, gen):
        # add new result
        if self.res == None:
            self.res = cur_res
            self.time_end = time.time()
            self.res['gen'] = [gen]
            self.res['run_time'] = ["{:.4f} sec".format(
                self.time_end - self.time_start)]
        if cur_res['cost'][0] < self.res['cost'][-1]:  # Better solution
            self.time_end = time.time()
            self.res['cost'].extend(cur_res['cost'])
            self.res['query'].extend(cur_res['query'])
            if self.mode == 2:
                self.res['value'].extend(cur_res['value'])
            else:
                self.res['value1'].extend(cur_res['value1'])
                self.res['value2'].extend(cur_res['value2'])
            self.res["mode"]='pairwise' if self.mode==2 else '3-seq'
            self.res['gen'].append(gen)
            self.res['run_time'].append(
                "{:.4f} sec".format(self.time_end-self.time_start))

    def optimize(self):
        """
        The critical function of GA
        Selects a group and generates new generation
        """
        print("Optimization starts!")
        for gen in tqdm(range(self.cfg.GA.MAX_GEN)):
            # print('Generation: {}'.format(gen))
            self.eval_ppl()
            best_traits = self.gene2traits(self.select_best().gene)
            worst_traits = self.gene2traits(self.select_worst().gene)
            cur_res = self.cost_traits(best_traits)
            worst_res = self.cost_traits(worst_traits)

            self.add_res(cur_res, gen)
            json.dump(self.res, open(osp.join(self.cfg.RESULT.DIR,
                                              'GA.json'), 'w'))
            # print("min_cost: {}".format(self.res['cost'][-1]))
            # print("max_cost: {}".format(worst_res['cost'][0]))
            chosen = self.selection()
            next_gen = []
            while len(next_gen) < self.cfg.GA.NEXT_GEN:
                pair = [chosen.pop() for _ in range(2)]
                if random.random() < self.cfg.GA.CROSSOVER.PROB:
                    new_pair = self.crossover(pair)
                    next_gen.append(new_pair[0])
                    next_gen.append(new_pair[1])
                else:
                    next_gen.extend(pair)
            self.gene_ppl = next_gen
        pprint(self.res)
