import random
from random import randint as rint
import json
import math
import numpy as np
from tqdm import tqdm
import os
import os.path as osp


class Gene:
    def __init__(self, traits):
        """
        data: the data we use to compute the fittness
        gene: the gene list
        length: length of gene
        fittness: fittness of this individual
        """
        self.traits = traits
        self.gene = self.traits2gene(traits)
        self.length = len(self.gene)
        self.fittness = None

    def __lt__(self, other):
        return self.fittness < other.fittness

    def traits2gene(self, traits):
        gene = []
        for t in traits:
            gene += t
        return gene


class GA:
    def __init__(self, cfg=None, loader=None, cost_func=None):
        self.loader = loader
        self.cfg = cfg
        self.cost_func = cost_func
        self.init_ppl()
        self.min_cost = math.inf

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
        print("Evaluating population...")
        trait_ppl = [self.gene2traits(g.gene) for g in self.gene_ppl]
        for i in range(len(self.gene_ppl)):  # for evry individual
            individual_cost = []
            # query
            q = self.loader.query[1]
            q = list(q)
            for j in range(len(trait_ppl[i][0])):
                q.insert(trait_ppl[i][0][j] + j, '-')
            q = ''.join(q)
            # database
            for j in range(len(self.loader.database)):  # for every seq
                s = self.loader.database[j]
                s = list(s)
                for k in range(len(trait_ppl[i][j + 1])):  # every pos
                    s.insert(trait_ppl[i][j + 1][k] + k, '-')
                s = ''.join(s)
                individual_cost.append(self.cost_func(q, s))
            self.gene_ppl[i].fittness = max([1000/(cost + 1e-8)  # avoid divide by 0
                                             for cost in individual_cost])

    def crossover(self, gene_pair):
        """
        This function generate two offsprings of the pair of Gene
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

    def mutation(self, gene: Gene):
        """
        This function randomly changes one element in gene
        bound: the pair of lower and upper bound of mutation element
        """
        bound = [1, max(gene)]
        length = len(gene)
        pos = random.randrange(0, length)
        gene[pos] = random.randint(bound[0], bound[1])
        return gene

    def selection(self):
        """
        This function randomly selects k individuals from the population
        individuals with higher fittness are more likely to be chosen
        """
        s_popu = sorted(self.gene_ppl, reverse=True)
        sum_fittness = sum(p.fittness for p in self.gene_ppl)
        print("Best fittness: {}".format(
            max(p.fittness for p in self.gene_ppl)))
        print("Mean fittness: {}".format(sum_fittness/len(self.gene_ppl)))
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
        This function selects the best individual
        """
        return max(self.gene_ppl)

    def cost_best(self):
        """
        Computes the cost of currently best individual
        """
        ans = {}
        best_traits = self.gene2traits(self.select_best().gene)
        ans['cost'] = math.inf
        q = self.loader.query[1]
        q = list(q)
        for j in range(len(best_traits[0])):
            q.insert(best_traits[0][j] + j, '-')
        q = ''.join(q)
        for j in range(len(self.loader.database)):  # for every seq
            s = self.loader.database[j]
            s = list(s)
            for k in range(len(best_traits[j + 1])):  # every pos
                s.insert(best_traits[j + 1][k] + k, '-')
            s = ''.join(s)
            cur_cost = self.cost_func(q, s)
            if cur_cost < ans['cost']:
                ans['cost'] = cur_cost
                ans['query'] = q
                ans['value'] = s
        return ans

    def optimize(self):
        """
        The critical function of GA
        Selects a group and generates new generation
        """
        print("Optimization starts!")
        for gen in tqdm(range(self.cfg.GA.MAX_GEN)):
            self.eval_ppl()
            print('Generation: {}'.format(gen))
            res = self.cost_best()
            if res['cost']<self.min_cost:# Better solution
                self.min_cost=res['cost']
                res['gen']=gen
                json.dump(res,open(osp.join(self.cfg.RESULT.DIR,\
                    'GA_PPL{}.json'.format(self.cfg.GA.PPL)),'w'))
            print("min_cost: {}".format(self.min_cost))
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
