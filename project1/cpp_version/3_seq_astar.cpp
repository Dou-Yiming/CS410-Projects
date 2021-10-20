#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <set>
#include <time.h>
#include <queue>
#include<math.h>
using namespace std;

struct res
{
    int cost;
    string q, v1, v2;
    res()
    {
        this->cost = 1000000;
        this->q = "";
        this->v1 = "";
        this->v2 = "";
    }
};

struct Node
{
    int pos[3];
    int g, f;
    string query, seq1, seq2;
    Node(int pos[3], int g, int f, string q = "", string s1 = "", string s2 = "")
    {
        this->pos[0] = pos[0];
        this->pos[1] = pos[1];
        this->pos[2] = pos[2];
        this->g = g;
        this->f = f;
        this->query = q;
        this->seq1 = s1;
        this->seq2 = s2;
    }
    bool operator<(const Node &other) const
    {
        return f < other.f;
    }
    bool operator>(const Node &other) const
    {
        return f > other.f;
    }
};

int alpha(char c1, char c2)
{
    return c1 == c2 ? 0 : 3;
}

int H_dist(int pos1[3], int pos2[3])
{
    vector<int> ans;
    for (int i = 0; i < 3; ++i)
        ans.push_back(abs(pos1[i] - pos2[i]));
    return (*max_element(ans.begin(), ans.end()) - *min_element(ans.begin(), ans.end())) * 4;
}

int M_dist(int pos1[3], int pos2[3])
{
    int ans = 0;
    for (int i = 0; i < 3; ++i)
        ans += abs(pos1[i] - pos2[i]);
    return ans * 1000;
}

bool is_equal(int a[3], int b[3])
{
    for (int i = 0; i < 3; ++i)
    {
        if (a[i] != b[i])
            return false;
    }
    return true;
}

int min(int a[3])
{
    int tmp = a[0] < a[1] ? a[0] : a[1];
    return a[2] < tmp ? a[2] : tmp;
}

int align_2_seq(string X, string Y, int delta)
{
    int m = X.length(), n = Y.length();
    vector<vector<int>> dp(m + 1);
    for (int i = 0; i <= m; ++i)
        dp[i].resize(n + 1);
    // init dp
    for (int i = 0; i <= m; ++i)
        dp[i][0] = 2 * i;
    for (int i = 0; i <= n; ++i)
        dp[0][i] = 2 * i;
    for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= n; ++j)
        {
            char c1 = X[i - 1], c2 = Y[j - 1];
            int cand[3] = {
                dp[i - 1][j - 1] + alpha(c1, c2),
                dp[i - 1][j] + delta,
                dp[i][j - 1] + delta,
            };
            dp[i][j] = min(cand);
        }
    return dp[m][n];
}

res align_3_seq(string X, string Y, string Z, int delta)
{
    // pre-compute as heuristic function
    int m = int(X.length()), n = int(Y.length()), o = int(Z.length());
    vector<vector<int>> matXY(m + 1), matYZ(n + 1), matXZ(m + 1);
    for (int i = 0; i <= m; ++i)
    {
        matXY[i].resize(n + 1);
        matXZ[i].resize(o + 1);
    }
    for (int i = 0; i <= n; ++i)
        matYZ[i].resize(o + 1);
    for (int i = 0; i <= m; ++i)
        for (int j = 0; j <= n; ++j)
            matXY[i][j] = align_2_seq(X.substr(i, m), Y.substr(j, n), delta);
    for (int i = 0; i <= m; ++i)
        for (int j = 0; j <= o; ++j)
            matXZ[i][j] = align_2_seq(X.substr(i, m), Z.substr(j, o), delta);
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= o; ++j)
            matYZ[i][j] = align_2_seq(Y.substr(i, n), Z.substr(j, o), delta);
    int goal[3] = {m, n, o};
    priority_queue<Node, vector<Node>, greater<Node>> open_list;

    int p[3] = {0, 0, 0};
    Node node(p, 0, 0);
    open_list.push(node);
    vector<vector<int>> closed_list;
    while (open_list.size() != 0)
    {
        Node head = open_list.top();
        open_list.pop();
        vector<int> pos;
        for (int i = 0; i < 3; ++i)
            pos.push_back(head.pos[i]);
        if (find(closed_list.begin(), closed_list.end(), pos) != closed_list.end())
            continue;
        else
            closed_list.push_back(pos);
        if (is_equal(head.pos, goal))
        {
            res r;
            r.cost = head.g;
            r.q = head.query;
            r.v1 = head.seq1;
            r.v2 = head.seq2;
            return r;
        }
        int p0 = head.pos[0], p1 = head.pos[1], p2 = head.pos[2];
        int neighbours[7][3] = {
            {p0 + 1, p1 + 1, p2 + 1},
            {p0 + 1, p1 + 1, p2},
            {p0 + 1, p1, p2 + 1},
            {p0, p1 + 1, p2 + 1},
            {p0 + 1, p1, p2},
            {p0, p1 + 1, p2},
            {p0, p1, p2 + 1}};
        for (int l = 0; l < 7; ++l)
        {
            int nei[3] = {neighbours[l][0], neighbours[l][1], neighbours[l][2]};
            vector<int> nei_vec;
            for (int i = 0; i < 3; ++i)
                nei_vec.push_back(nei[i]);
            if (find(closed_list.begin(), closed_list.end(), nei_vec) != closed_list.end() || nei[0] > goal[0] || nei[1] > goal[1] || nei[2] > goal[2])
                continue;
            int g = head.g, h, f;
            int cost_added;
            string q, s1, s2;
            if (l == 0)
            {
                cost_added = alpha(
                                 X[p0], Y[p1]) +
                             alpha(
                                 Y[p1], Z[p2]) +
                             alpha(X[p0], Z[p2]);
                h = matXY[p0 + 1][p1 + 1] + matXZ[p0 + 1][p2 + 1] + matYZ[p1 + 1][p2 + 1];
                q = head.query + X[p0];
                s1 = head.seq1 + Y[p1];
                s2 = head.seq2 + Z[p2];
            }
            else if (l == 1)
            {
                cost_added = alpha(
                                 X[p0], Y[p1]) +
                             2 * delta;
                h = matXY[p0 + 1][p1 + 1] + matXZ[p0 + 1][p2] + matYZ[p1 + 1][p2];
                q = head.query + X[p0];
                s1 = head.seq1 + Y[p1];
                s2 = head.seq2 + "-";
            }
            else if (l == 2)
            {
                cost_added = alpha(
                                 X[p0], Z[p2]) +
                             2 * delta;
                h = matXY[p0 + 1][p1] + matXZ[p0 + 1][p2 + 1] + matYZ[p1][p2 + 1];
                q = head.query + X[p0];
                s1 = head.seq1 + "-";
                s2 = head.seq2 + Z[p2];
            }
            else if (l == 3)
            {
                cost_added = alpha(
                                 Y[p1], Z[p2]) +
                             2 * delta;
                h = matXY[p0][p1 + 1] + matXZ[p0][p2 + 1] + matYZ[p1 + 1][p2 + 1];
                q = head.query + "-";
                s1 = head.seq1 + Y[p1];
                s2 = head.seq2 + Z[p2];
            }
            else if (l == 4)
            {
                cost_added = 2 * delta;
                h = matXY[p0 + 1][p1] + matXZ[p0 + 1][p2] + matYZ[p1][p2];
                q = head.query + X[p0];
                s1 = head.seq1 + "-";
                s2 = head.seq2 + "-";
            }
            else if (l == 5)
            {
                cost_added = 2 * delta;
                h = matXY[p0][p1 + 1] + matXZ[p0][p2] + matYZ[p1 + 1][p2];
                q = head.query + "-";
                s1 = head.seq1 + Y[p1];
                s2 = head.seq2 + "-";
            }
            else if (l == 6)
            {
                cost_added = 2 * delta;
                h = matXY[p0][p1] + matXZ[p0][p2 + 1] + matYZ[p1][p2 + 1];
                q = head.query + "-";
                s1 = head.seq1 + "-";
                s2 = head.seq2 + Z[p2];
            }
            g += cost_added;
            f = g + h;
            Node new_node(nei, g, f, q = q, s1 = s1, s2 = s2);
            open_list.push(new_node);
        }
    }
    return res();
}

int main()
{
    clock_t timer_start = clock();
    // read data
    ifstream query_in("./query.txt", ifstream::in);
    ifstream db_in("../data/MSA_database.txt", ifstream::in);
    if (!query_in.is_open() || !db_in.is_open())
    {
        cerr << "Error: cannot open input file" << endl;
        exit(-1);
    }
    string query, seq;
    vector<string> db;
    query_in >> query;
    while (getline(db_in, seq))
        db.push_back(seq);
    query_in.close();
    db_in.close();
    // search
    res final_res;
    int delta = 2;
    for (int i = 0; i < db.size(); ++i)
        for (int j = i + 1; j < db.size(); ++j)
        {
            string Y = db[i];
            string Z = db[j];
            res cur_res = align_3_seq(query, Y, Z, delta);
            final_res = cur_res.cost < final_res.cost ? cur_res : final_res;
        }
    printf("min_cost: %d \n", final_res.cost);
    cout << final_res.q << endl;
    cout << final_res.v1 << endl;
    cout << final_res.v2 << endl;
    clock_t timer_end = clock();
    cout << "Running time: " << (timer_end - timer_start) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
}