## Project1: Implement three algorithms to solve multiple sequence alignment (MSA) problems

In this report, the implementation, results, running time and time complexity of the 3 algorithms will be described respectively.

### Implementation & Time Complexity

#### Dynamic Programming

#### A-star (A*)

#### Genetic Algorithm

### Results & Running time

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

1. query: KJXXJAJKPXKJJXJKPXKJXXJAJKPXKJJXJKPXKJXXJAJKPXKJXXJAJKHXKJXXJAJKPXKJXXJAJKHXKJXX
    - DP & A*:
        - KJX---XJAJKPXKJ-JX-JKPXKJXXJA--J-KPXKJ-JX-JKPXKJXXJAJKPXKJ-XXJAJ--KH--X-KJ-XXJAJKPXKJXXJAJKHXKJXX--
        - --XHAPXJAJ-XXXJAJXDJAJX--XXJAPXJAHXXXJAJXDJAJX--XXJAJ-XXXJPPXJAJXXXHAPXJAJXXXJAJ--X--XXJAJ--X--XXJA
    - Genetic Algorithm:
        - -K-JX-X-JA-J-K-P-XK-J---J-X-J-KPX-KJXXJA-JK-PX-K-JJ---XJ-KPX-KJX-X-J--A-JK--PXKJ-XXJA-J-KHXKJX--XJAJ-KPXKJ--XXJAJKHXKJXX
        - P-PJ--J-P-J-J--P-P--J-JPP-J-J-K---JJPPJ-JPP-JJ-G-T-J---J--J-LJJM--TJNL--J--J--T--T-JU-J---H-OJ---PGJ-JP-PJ-J--PJ-UPHJJ-P
2. query: ILOTGJJLABWTSTGGONXJMUTUXSJHKWJHCTOQHWGAGIWLZHWPKZULJTZWAKBWHXMIKLZJGLXBPAHOHVOLZWOSJJLPO
    - DP & A*:
        - ILOTGJJLABWTSTG-GONXJ---M--UTUXSJHKWJHCTOQ-HWGAGIWLZHWPKZULJTZWAKBWHXMIKLZJGLXBPAHOHVOLZWOSJJLP-O
        - IPOTWJJLAB-KS-GZGW-JJKSPPOPHT--SJE-WJHCTOOTH-RAXBKLBHWPKZULJPZKAKVKHXUIKLZJGLXBTGHOHBJLZPPSJJ-PJO
    - Genetic Algorithm:
        - ILOTGJJ-L-ABW-TST-G-GO-NXJM-U-TUX--SJHKWJ-HC-T-O-Q-HW-GAGI-WLZHWPKZULJ--T-ZWAK--B-WHXMI-KLZJGLXB-PAHO-H--VO-L-ZWOSJ-JLPO
        - IP-OTWJ-JLABK--SG--ZGW-J-JKSPP-OPHTSJEWJH-CTOO-TH--RA--XB--KLBHWP-K-ZULJ--PZKA-KV-KHXUI-KLZJGLX-BTG-H-O-HBJL-ZPPS-JJP-JO
3. query: IHKKKRKKKKKKXGWGKKKPKSKKKKKBKKKPKHKKXKKBSKKPKWKKLKSKRKKWXKPKKBKKKPKTSKHKKKKLADKKYPKKKOPHKKBWWLPPWKK
    - DP & A*:
        - IHKKKRKKKKKKXGWGKKKPKSKKKK--KBKKKP-KHKKXKK--BSK-K-PKWKKLKSKRKKWXKPKK-BK-KKPKTSKHKKKKLADKKYPKKKOPHKKBWWLPPWKK-
        - I---K-BSKKKK--W-KKK-K-KKKKWWK-KKKPGK-KK-KKXXGGKRKWWKWKKPK-K-KKKXK-KKRWKMKKPK--KWPKKK---KK-PGKK---KLBKW---WKKJ
    - Genetic Algorithm:
        - IH-K-K-KRK-KKKKKXGWGKKK--PKS-KKKK-KBKKK-PKHK-KXKK-B-S-KKPKWK-KLKSKRKKWXKPKKBKKKPKTSKHKKKKLADKKY-P-KK--K-OPHKKB-WWLPPWK-K
        - I-KBSK-K-K-K-WKKK-K-KKK-KWWK--KK--KPG--KKKK-K-XXGG-KR-KW--WKWKK-P--KK--K-KKXK-KK-RWKMKKPKKWP-K--KKKKP-G-K-K-K-LBK-W-WKKJ
4. query: MPPPJPXPGPJPPPXPPPJPJPPPXPPPPSPPJJJPPXXPPPPPJPPPXPPXIPJMMMXPKPSVGULMHHZPAWHTHKAAHHUPAONAPJSWPPJGA
    - DP & A*:
        - MPPPJPX-PGP-JPPPXPPPJPJPPPXPPPPSPPJJJPPXXP-PPPPJPPPXPPXIPJMMMXPKPSVGULMHHZPAWHTHKAAHHUPAONAP-JSWPP-JGA
        - --OPJPXJP-PMJPPMX-PP-MJ-PPXJPPOXPPXJJPJXXPXJPPOJPPMXPPOGP--PXXP-P----OM---PPXXPPOXPPXJP-QXPPBJ-PPPXPPX
    - Genetic Algorithm:
        - MPPPJPXPGPJPPPXP--PPJ-PJPP-PXPP-PPSPPJJ-J-PPXXPP--P-PPJPP-P-XPPX--IPJMMMXPKPS-VGULMHH-ZPA-WHTH-KAAH-H-UPA-O-NAPJ-SWPPJGA
        - --P-JJ-J--A--PJ--J-J-APJ-J-K-PJ-HP-PPJJ---PH-A-J-J--PP-P-J-J--L-PP-J--JPH--P-J-JL--PP-J---J--P-PPJJ-L--PPJ--JPPJ-J---J-T
5. query: IPPVKBKXWXKHSAPHVXXVOJMRAKKPJVLLJBWKOLLJKXHGXLLCPAJOBKPGXBATGXMPOMCVZTAXVPAGKXGOMJQOLJGWGKXLQ
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
| A-star (A*)         | ①: 290<br />②: 122<br />   | ①: ~20h (Python)<br />②: ~20h (Python)<br />                 |
| Genetic Algorithm   | ①: 609<br />②: 600<br />   | ①: 2min45s<br />②: 55.67s<br />                              |

**The alignment of each query found by each algorithm:**

1. query: IPZJJLMLTKJULOSTKTJOGLKJOBLTXGKTPLUWWKOMOYJBGALJUKLGLOSVHWBPGWSLUKOBSOPLOOKUKSARPPJ
    - DP & A*:
        - IPZJJ-LMLTKJULOSTKTJOGLKJOBLTXGKTPLUWWKOMOYJBGALJUKLGLOSVHWBPGWS-LUKOBSOPLOOKU---KSARPPJ
        - IPZJJPL-LTHUULOSTXTJOGLKJGBLLMMPJPLUWGKOMOYJBZA-YUKOFLOSZHGBP-HXPLXKJBXKJL-AUUOJHW-TWWPQ
        - IPMJJ-LLLTHOULOSTMAJIGLKJPVLLXGKTPLTWWKOMOYJBZP-YUKLILOSZHGBPGWX-LZKJBSWJLPJUU-MHK-TRAP-
    - Genetic Algorithm:
        - -IPZ--J-JLM-LT-KJUL-O-STKTJO--G-LKJ-OBL-TX-GK-TPL-UWWK-OM-OYJ-B-G-ALJ--UKLGLOSV-HWBP-G--WSLU-KOB-SOP--LO-OKU-K-S-A--RPPJ
        - -IPP--J-OJPJJ--JP--JO-J-P-P--A--J--O----J--P-K-JO-JP---JJJP---P-AJOJ--PJ-JJ-P-PAJ---O-J----PJJ-JP-P-A-J-O-J-P-JJ-----J-P
        - -IP-H-JJJ-PJJJ-P-A-JO--J-PJJ-JPAJOJP-P-J-J---JK-H-JJ-JP-JJ--J--P-JJ-J--PAJOJP-J-J-JPHJJ-J--PJ-JJP--A-J-O--JPJJJ--P--HJ-J
2. query: IWTJBGTJGJTWGBJTPKHAXHAGJJSJJPPJAPJHJHJHJHJHJHJHJHJPKSTJJUWXHGPHGALKLPJTPJPGVXPLBJHHJPKWPPDJSG
    - DP & A*:
        - --IWTJBGTJGJTWGBJTPKHAXHAGJJSJJPPJAPJHJHJHJHJHJHJHJHJPKSTJJUWXHGPHGALKLPJTPJPGVXPLBJHHJPKWPPDJSG
        - --IWTJBGTJGJTWGBJTPKHAXHAGJJXJJKPJTPJHJHJHJHJHJHJHJHJHKUTJJUWXHGHHGALKLPJTPJPGVXPLBJHH----------
        - WPIWTJBGTJGJTHGBJOXKHTXHAGJJXJJPPJTP--JHJHJHJHJHJHJHJPKUAJJUWXHGHHGALKLPJTPJPGVXPLBJHHJPK-------
    - Genetic Algorithm:
        - I-WTJB-G-TJGJTW-G-BJT-P-KHAXHA--GJJ-SJJP--P-JAPJ-HJH-JHJHJHJHJHJHJPKS-TJJUWXHGPH--GAL-K-L-PJTPJ-PGV--XPL-BJHH-JPKWPPDJSG
        - --IP-P-J-O--J-PJ-J-JPJ-OJP-P-A--J--O-J-PK---JO---JP-J-JJP-P-AJ-O-J-P-J--JJ-P--P-A--J--O---J--PJ--JJ--PP-A-J--OJ---PJJJ-P
        - --IPH--JJ-JPJ-JJ-PAJOJ-PJJ--J-PAJO--J-PPJ--JJ-K-H--J-JJ--PJ--J---JP-JJJPAJO--J-P-JJ--J--PH-J-JJ-PJJJ-PAJO-J--PJ-J-JP-HJJ