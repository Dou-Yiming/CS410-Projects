# Project1: Implement three algorithms to solve multiple sequence alignment (MSA) problems

In this document, the implementation, results, running time and time complexity of the 3 algorithms will be described respectively.

## Implementation & Time Complexity

### Dynamic Programming

#### **Implementation:**

DP algorithm can be regarded as searching on a $k$-dimensional space, and the $2$-dim situation can be illustrated in the following picture:

<img src="D:\OneDrive - sjtu.edu.cn\大三上\人工智能\Projects\project1\README.assets\image-20211004155158303.png" alt="image-20211004155158303" style="zoom:50%;" />

In this situation, the element of the dp can be computed by:$dp[i][j]=min(dp[i-1][j-1]+\alpha_{x_i,y_j},dp[i-1][j]+\delta,dp[i][j-1]+\delta)$

The execution of DP algorithm can be described by the following steps:

1. Initialize the lines or surfaces of dp:

    ```python
    # pairwise
    # 2 axis
    for i in range(dp.shape[0]):
    	dp[i][0] = 2*i
    for i in range(dp.shape[1]):
    	dp[0][i] = 2*i
        
    # 3-sequence
    # axis X 
    for i in range(dp.shape[0]):
    	dp[i][0][0] = 4*i # axis Y and Z should be initialized similarly
    # 3 surfaces
    # XY:
    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):
            c1, c2 = X[i-1], Y[j-1]
            cand = [
                dp[i-1][j-1][0] + self.alpha(c1, c2) + 2 * self.delta,
                dp[i-1][j][0] + 2 * self.delta,
                dp[i][j-1][0] + 2 * self.delta
            ]
            dp[i][j][0] = min(cand)
    # surface YZ and XZ should be initialized similarly
    ```

    For the pairwise alignment, the axis-x and axis-y should be initialized.

    For the 3-seq alignment, the axix-x (y,z) and surfaces (XY, YZ, XZ) should all be initialized. 

2. Search:

    ```python
    # pairwise
    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):
            c1, c2 = X[i-1], Y[j-1]
            cand = [dp[i-1][j-1] + self.alpha(c1, c2),
                    dp[i-1][j] + self.delta,
                    dp[i][j-1] + self.delta,
                    ]
            dp[i][j] = min(cand)
    # 3-sequence
    for i in range(1, dp.shape[0]):  # query
        for j in range(1, dp.shape[1]):  # seq1
            for k in range(1, dp.shape[2]):  # seq2
                c1, c2, c3 = X[i-1], Y[j-1], Z[k-1]
                cand = [
                    dp[i-1][j-1][k-1] + self.alpha(c1, c2) + self.alpha(c2, c3) + self.alpha(c1, c3),
                    dp[i-1][j-1][k] + self.alpha(c1, c2) + 2 * self.delta,
                    dp[i-1][j][k-1] + self.alpha(c1, c3) + 2 * self.delta,
                    dp[i][j-1][k-1] + self.alpha(c2, c3) + 2 * self.delta,
                    dp[i][j][k-1] + 2 * self.delta,
                    dp[i][j-1][k] + 2 * self.delta,
                    dp[i-1][j][k] + 2 * self.delta
                ]
                dp[i][j][k] = min(cand)
    ```

    For the pairwise alignment, the searching procedure is done on a surface. Therefore, each position is reachable from 3 conjunctive positions, thus the amount of candidates is 3.

    For the 3-seq alignment, the searching procedure is done in a cubic. Therefore, each position is reachable from 7 conjunctive positions, thus the amount of candidates is 7.

3. By the end of searching procedure, the last-viewed (right-bottom corner) element of the dp is the optimal result.

#### **Time Complexity**

To find the optimal solution, every position in the searching space should be visited. Suppose that the amount of nodes in the searching space is $n$

Therefore, the time complexity of $k$-seq alignment is $O(n)$.

Similarly, the space complexity is also $O(n)$.

#### **Optimization**

The space complexity is so large that make it impossible to find the optimal solution when the dimension is large.

However, according to the method introduced by *Hirschberg* in 1975 that combines divide-and-conquer and DP, the space can be reduced to linear space: $O(l[0]+l[1]+\cdots+l[k-1])$, in which $l$ is the length of each sequence.

This enables DP to solve much larger $k$-seq alignment problems.

### A-star (A*)

#### **Implementation**

The idea of A* search is that we should avoid expanding the paths that are expensive, which can be estimated by evaluation function: $f(n)=g(n)+h(n)$. $g(n)$ is the cost so far to reach $n$. $h(n)$ is the estimated cost from $n$ to goal, which is also called heuristic function.

For each iteration in the searching procedure, the node $n$ with the smallest evaluation cost will be visited, making the searching position to reach goal faster.

While $g(n)$ is decided by the position, the design of heuristic function can be varied. Moreover, different designs of heuristic function may significantly influence the performance, making it essential to design a better heuristic function carefully.

In this project, I design 2 different heuristic functions for pairwise and 3-sequence alignment:

1. For the pairwise alignment, the estimated cost of node at $(x,y)$ is $abs((g_x-x)-(g_y-y))$. 

    Since the discovery path should return to the diagonal, this heuristic function never overestimates the cost to reach the goal, which means that we are sure to find the optimal solution using tree-search.

2. For the 3-seq alignment,  I referred to the given reference paper *Comparing Best-First Search and Dynamic Programming for Optimal Multiple  Sequence Alignment*, which introduced a better lower bound of the estimated cost.

    It is obvious that the cost of an optimal pairwise alignment is always less than or equal to the cost of any other alignment of the 2 strings as part of a multiple alignment. Therefore, for 3-seq alignment, we can pre-compute the cost of aligning suffixes of each pair of strings, and the heuristic cost of any node in the cube is the sum of the corresponding 3 costs of pairwise alignments.

    Similarly, this function also never overestimates the cost to reach the goal, thus being optimal.

#### **Time Complexity**

The heuristic function reduces the amount of nodes that must be visited to find the solution. However, it is impossible to determine to which extent does the heuristic function reduce the amount. Therefore, only the upper bound of the time complexity can be estimated, in which we hypothesize that the reduced amount is $0$.

Suppose the amount of all nodes in the space is $n$. 

Based on the hypothesis that the reduced amount is $0$, every node should be popped from the open-list in the outer loop, thus needing to execute $n$ times of outer loop.

In the inner loop, the conjunctive nodes of the popped node should be visited and pushed into the open-list.

Based on the analysis above, each node is pushed into and popped from the open-list for one time. Pay attention that the open-list in my implementation is a **min-heap**, which is an tiny optimization. Therefore, the overall time complexity is $O(nlogn)$.

### Genetic Algorithm

#### Implementation

**Encoding Method**

In order to transform each solution into the gene of an individual in the genetic algorithm, an encoding method should be figured out. 

The encoding method must satisfy the following rules: 

1. Each feasible scheduling method has a one-to-one correspondence with its gene. 
2. Genes are allowed to mutate without losing feasibility. 
3. A pair of genes can crossover with each other and generate new genes of feasible solutions.

To satisfy the rules above, we encode each solution by the following procedure: 

1. For each sequence, create an array representing the location of inserting "-".
2. Concatenate the arrays of all sequences into a long array, and this array is the gene of the alignment method.

**Optimization**

In order to reach the goal that minimizes the alignment cost, the cost of alignment determined by each gene is computed in each iteration

In the process of crossover and mutation, the individuals with larger fitness are chosen to generate the next generation, and the overall fitness will increase over time.

#### Time Complexity

The complexity of GA is the complexity of the optimization procedure.

Before the analysis, we define the meanings of symbols:

|   Symbol   |                 Definition                 |
| :--------: | :----------------------------------------: |
|     G      |             generation number              |
|     P      |             population amount              |
|   $L_g$    |                gene length                 |
| $L_{seq}$  |          longest sequence length           |
| $T_{func}$ | evaluation time of an individual's fitness |
| $T_{opt}$  |    total time of optimization procedure    |

The time complexity is analyzed as following:

- $P\times T_{func}$ represents the time used to evaluate the whole population
- $P\times logP+P^2$ is the time of sorting the individuals and select those with higher fitness
- $P\times L$ is the time needed to crossover and generate the new generation.
- The fitness computation is $O(L_{seq})$

All in all, the time complexity of GA is:
$$
T_{opt}=O(G\times[P\times L_{seq}+(P\times logP+P^2)+P\times L_g])
$$

- When $P$ is relatively large, $T_{opt}=O(G\times P^2)$
- When $E$ is relatively large, $T_{opt}=O(P\times E)$
- When $L_g$ is relatively large, $T_{opt}=O(P\times L_{g})$

## Results & Running time

This section shows the results of the alignment of each algorithm.

All of the experiments are performed on my laptop with an *Intel i7-10875H@2.3GHz* CPU and 16GB memory.

The population amount of Genetic Algorithm is 100, while the probability of crossover and mutation is 0.4 and 0.6, respectively.

#### Results of Pairwise Alignment:

| Algorithm           | Result (cost of alignment)                                   | Running Time                                                 |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Dynamic Programming | ①: 112<br />②: 107<br />③: 113<br />④: 148<br />⑤: 131<br /> | ①: 1.82s<br />②: 2.06s<br />③: 2.17s<br />④: 2.27s<br />⑤: 2.24s<br /> |
| A-star (A*)         | ①: 112<br />②: 107<br />③: 113<br />④: 148<br />⑤: 131<br /> | ①: 2min10s<br />②: 2min38s<br />③: 3min58s<br />④: 3min40s<br />⑤: 3min9s<br /> |
| Genetic Algorithm   | ①: 188 <br />②: 208<br />③: 184<br />④: 206<br />⑤: 219<br /> | ①: 0.60s<br />②: 0.57s<br />③: 0.92s<br />④: 2.78s<br />⑤: 1.58s<br /> |

**The alignment of each query found by each algorithm:**

1. **query:** KJXXJAJKPXKJJXJKPXKJXXJAJKPXKJJXJKPXKJXXJAJKPXKJXXJAJKHXKJXXJAJKPXKJXXJAJKHXKJXX
    - DP & A*:
        - KJX---XJAJKPXKJ-JX-JKPXKJXXJA--J-KPXKJ-JX-JKPXKJXXJAJKPXKJ-XXJAJ--KH--X-KJ-XXJAJKPXKJXXJAJKHXKJXX--
        - --XHAPXJAJ-XXXJAJXDJAJX--XXJAPXJAHXXXJAJXDJAJX--XXJAJ-XXXJPPXJAJXXXHAPXJAJXXXJAJ--X--XXJAJ--X--XXJA
    - Genetic Algorithm:
        - -K-JX-X-JA-J-K-P-XK-J---J-X-J-KPX-KJXXJA-JK-PX-K-JJ---XJ-KPX-KJX-X-J--A-JK--PXKJ-XXJA-J-KHXKJX--XJAJ-KPXKJ--XXJAJKHXKJXX
        - P-PJ--J-P-J-J--P-P--J-JPP-J-J-K---JJPPJ-JPP-JJ-G-T-J---J--J-LJJM--TJNL--J--J--T--T-JU-J---H-OJ---PGJ-JP-PJ-J--PJ-UPHJJ-P
2. **query:** ILOTGJJLABWTSTGGONXJMUTUXSJHKWJHCTOQHWGAGIWLZHWPKZULJTZWAKBWHXMIKLZJGLXBPAHOHVOLZWOSJJLPO
    - DP & A*:
        - ILOTGJJLABWTSTG-GONXJ---M--UTUXSJHKWJHCTOQ-HWGAGIWLZHWPKZULJTZWAKBWHXMIKLZJGLXBPAHOHVOLZWOSJJLP-O
        - IPOTWJJLAB-KS-GZGW-JJKSPPOPHT--SJE-WJHCTOOTH-RAXBKLBHWPKZULJPZKAKVKHXUIKLZJGLXBTGHOHBJLZPPSJJ-PJO
    - Genetic Algorithm:
        - ILOTGJJ-L-ABW-TST-G-GO-NXJM-U-TUX--SJHKWJ-HC-T-O-Q-HW-GAGI-WLZHWPKZULJ--T-ZWAK--B-WHXMI-KLZJGLXB-PAHO-H--VO-L-ZWOSJ-JLPO
        - IP-OTWJ-JLABK--SG--ZGW-J-JKSPP-OPHTSJEWJH-CTOO-TH--RA--XB--KLBHWP-K-ZULJ--PZKA-KV-KHXUI-KLZJGLX-BTG-H-O-HBJL-ZPPS-JJP-JO
3. **query:** IHKKKRKKKKKKXGWGKKKPKSKKKKKBKKKPKHKKXKKBSKKPKWKKLKSKRKKWXKPKKBKKKPKTSKHKKKKLADKKYPKKKOPHKKBWWLPPWKK
    - DP & A*:
        - IHKKKRKKKKKKXGWGKKKPKSKKKK--KBKKKP-KHKKXKK--BSK-K-PKWKKLKSKRKKWXKPKK-BK-KKPKTSKHKKKKLADKKYPKKKOPHKKBWWLPPWKK-
        - I---K-BSKKKK--W-KKK-K-KKKKWWK-KKKPGK-KK-KKXXGGKRKWWKWKKPK-K-KKKXK-KKRWKMKKPK--KWPKKK---KK-PGKK---KLBKW---WKKJ
    - Genetic Algorithm:
        - IH-K-K-KRK-KKKKKXGWGKKK--PKS-KKKK-KBKKK-PKHK-KXKK-B-S-KKPKWK-KLKSKRKKWXKPKKBKKKPKTSKHKKKKLADKKY-P-KK--K-OPHKKB-WWLPPWK-K
        - I-KBSK-K-K-K-WKKK-K-KKK-KWWK--KK--KPG--KKKK-K-XXGG-KR-KW--WKWKK-P--KK--K-KKXK-KK-RWKMKKPKKWP-K--KKKKP-G-K-K-K-LBK-W-WKKJ
4. **query:** MPPPJPXPGPJPPPXPPPJPJPPPXPPPPSPPJJJPPXXPPPPPJPPPXPPXIPJMMMXPKPSVGULMHHZPAWHTHKAAHHUPAONAPJSWPPJGA
    - DP & A*:
        - MPPPJPX-PGP-JPPPXPPPJPJPPPXPPPPSPPJJJPPXXP-PPPPJPPPXPPXIPJMMMXPKPSVGULMHHZPAWHTHKAAHHUPAONAP-JSWPP-JGA
        - --OPJPXJP-PMJPPMX-PP-MJ-PPXJPPOXPPXJJPJXXPXJPPOJPPMXPPOGP--PXXP-P----OM---PPXXPPOXPPXJP-QXPPBJ-PPPXPPX
    - Genetic Algorithm:
        - MPPPJPXPGPJPPPXP--PPJ-PJPP-PXPP-PPSPPJJ-J-PPXXPP--P-PPJPP-P-XPPX--IPJMMMXPKPS-VGULMHH-ZPA-WHTH-KAAH-H-UPA-O-NAPJ-SWPPJGA
        - --P-JJ-J--A--PJ--J-J-APJ-J-K-PJ-HP-PPJJ---PH-A-J-J--PP-P-J-J--L-PP-J--JPH--P-J-JL--PP-J---J--P-PPJJ-L--PPJ--JPPJ-J---J-T
5. **query:** IPPVKBKXWXKHSAPHVXXVOJMRAKKPJVLLJBWKOLLJKXHGXLLCPAJOBKPGXBATGXMPOMCVZTAXVPAGKXGOMJQOLJGWGKXLQ
    - DP & A*:
        - IPPVKBKXWXKHSA-PHVXXVOJMRAK--KPJVLLJBWKOLLJKXHGXL--LCPAJOBKPGXBATGXMPOMCVZTAXV-P--AGKXGOMJQO----LJGWGKXLQ
        - ITPVKWKSKXKXUAXP-VHXVO-M-MKHYBPABLLOBGKOLLJGXZGXLSOL--AMOGKIGXBATBXMPJTCVMTAXVMPWWA---WOM--OUPHHZBITKKXLK
    - Genetic Algorithm:
        - I-PPV-KBKX--W-XKHSA-P-HVXX-VOJ-MRAKKPJV--LLJ-BWKOL-LJKX-HG-XLL-C-P-A-JOB-KPGXB-ATGXM-POMCVZ-TAXVPAGKX-GO-M-JQOLJGW-GKXLQ
        - H-PJ-O-JKPJJ--P-H-H-P-OJ-P-K-J-J-P-K---J-J---PHP--J-JP-P-A-J-J-P-P--AJ--J-P--PJ-JP---KOJ-P-P-A--OJHK--J--JP-P-JJ-PP--O-J

#### Results of 3-seq Alignment:

| Algorithm           | Result (cost of alignment) | Running Time                                                 |
| ------------------- | -------------------------- | ------------------------------------------------------------ |
| Dynamic Programming | ①: 290<br />②: 122<br />   | ①: ~20h (Python),  4min59s (C++)<br />②: ~20h (Python),  4min38s (C++)<br /> |
| A-star (A*)         | ①: 290<br />②: 122<br />   | ①: >20h (Python), ~6h (C++)<br />②: >20h (Python), ~6h (C++)<br /> |
| Genetic Algorithm   | ①: 609<br />②: 600<br />   | ①: 2min45s<br />②: 55.67s<br />                              |

**The alignment of each query found by each algorithm:**

1. **query:** IPZJJLMLTKJULOSTKTJOGLKJOBLTXGKTPLUWWKOMOYJBGALJUKLGLOSVHWBPGWSLUKOBSOPLOOKUKSARPPJ
    - DP & A*:
        - IPZJJ-LMLTKJULOSTKTJOGLKJOBLTXGKTPLUWWKOMOYJBGALJUKLGLOSVHWBPGWS-LUKOBSOPLOOKU---KSARPPJ
        - IPZJJPL-LTHUULOSTXTJOGLKJGBLLMMPJPLUWGKOMOYJBZA-YUKOFLOSZHGBP-HXPLXKJBXKJL-AUUOJHW-TWWPQ
        - IPMJJ-LLLTHOULOSTMAJIGLKJPVLLXGKTPLTWWKOMOYJBZP-YUKLILOSZHGBPGWX-LZKJBSWJLPJUU-MHK-TRAP-
    - Genetic Algorithm:
        - -IPZ--J-JLM-LT-KJUL-O-STKTJO--G-LKJ-OBL-TX-GK-TPL-UWWK-OM-OYJ-B-G-ALJ--UKLGLOSV-HWBP-G--WSLU-KOB-SOP--LO-OKU-K-S-A--RPPJ
        - -IPP--J-OJPJJ--JP--JO-J-P-P--A--J--O----J--P-K-JO-JP---JJJP---P-AJOJ--PJ-JJ-P-PAJ---O-J----PJJ-JP-P-A-J-O-J-P-JJ-----J-P
        - -IP-H-JJJ-PJJJ-P-A-JO--J-PJJ-JPAJOJP-P-J-J---JK-H-JJ-JP-JJ--J--P-JJ-J--PAJOJP-J-J-JPHJJ-J--PJ-JJP--A-J-O--JPJJJ--P--HJ-J
2. **query:** IWTJBGTJGJTWGBJTPKHAXHAGJJSJJPPJAPJHJHJHJHJHJHJHJHJPKSTJJUWXHGPHGALKLPJTPJPGVXPLBJHHJPKWPPDJSG
    - DP & A*:
        - --IWTJBGTJGJTWGBJTPKHAXHAGJJSJJPPJAPJHJHJHJHJHJHJHJHJPKSTJJUWXHGPHGALKLPJTPJPGVXPLBJHHJPKWPPDJSG
        - --IWTJBGTJGJTWGBJTPKHAXHAGJJXJJKPJTPJHJHJHJHJHJHJHJHJHKUTJJUWXHGHHGALKLPJTPJPGVXPLBJHH----------
        - WPIWTJBGTJGJTHGBJOXKHTXHAGJJXJJPPJTP--JHJHJHJHJHJHJHJPKUAJJUWXHGHHGALKLPJTPJPGVXPLBJHHJPK-------
    - Genetic Algorithm:
        - I-WTJB-G-TJGJTW-G-BJT-P-KHAXHA--GJJ-SJJP--P-JAPJ-HJH-JHJHJHJHJHJHJPKS-TJJUWXHGPH--GAL-K-L-PJTPJ-PGV--XPL-BJHH-JPKWPPDJSG
        - --IP-P-J-O--J-PJ-J-JPJ-OJP-P-A--J--O-J-PK---JO---JP-J-JJP-P-AJ-O-J-P-J--JJ-P--P-A--J--O---J--PJ--JJ--PP-A-J--OJ---PJJJ-P
        - --IPH--JJ-JPJ-JJ-PAJOJ-PJJ--J-PAJO--J-PPJ--JJ-K-H--J-JJ--PJ--J---JP-JJJPAJO--J-P-JJ--J--PH-J-JJ-PJJJ-PAJO-J--PJ-J-JP-HJJ

## Results & Conclusion

1. DP and A* search are both optimal algorithm that can find the same optimal solution, while GA may not be able to find the optimal solution in limited time.

2. As for the 2, 3-seq alignment, DP performs much better than A* search.

    The reason is that the time complexity of DP is $O(n)$ while the upper bound of time complexity of A* is $O(nlog n)$. This means that if the heuristic function is not able to reduce the amount of nodes needed to be visited during the searching procedure, the $log n$ factor may pose large influence. 

    This phenomenon is common when the dimension is low. However, when the dimension is larger than $4$, A* algorithm will significantly outperforms DP, this conclusion is proved in the reference paper: *Comparing Best-First Search and Dynamic Programming for Optimal Multiple  Sequence Alignment*

## Supplementary

If you have any question for any part of this project, please feel free to contact me by the following methods:

E-mail: douyiming@sjtu.edu.cn

Wechat: 18017112986

