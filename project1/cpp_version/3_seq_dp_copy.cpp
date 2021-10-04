#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <set>
#include <time.h>
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
    res(int cost, string q, string v1, string v2)
    {
        this->cost = cost;
        this->q = q;
        this->v1 = v1;
        this->v2 = v2;
    }
};

int alpha(char c1, char c2)
{
    return c1 == c2 ? 0 : 3;
}

res DP(string &X, string &Y, string &Z, const int &delta = 2)
{
    // init dp
    int m = X.length();
    int n = Y.length();
    int o = Z.length();
    vector<vector<vector<int>>> dp(m + 1);
    for (int i = 0; i <= m; ++i)
    {
        dp[i].resize(n + 1);
        for (int j = 0; j <= n; ++j)
            dp[i][j].resize(o + 1);
    }
    for (int i = 0; i <= m; ++i)
        dp[i][0][0] = 2 * i;
    for (int i = 0; i <= n; ++i)
        dp[0][i][0] = 2 * i;
    for (int i = 0; i <= o; ++i)
        dp[0][0][i] = 2 * i;
    // store route
    vector<vector<vector<string>>> query(m + 1), value1(m + 1), value2(m + 1);
    for (int i = 0; i <= m; ++i)
    {
        query[i].resize(n + 1);
        value1[i].resize(n + 1);
        value2[i].resize(n + 1);
        for (int j = 0; j <= n; ++j)
        {
            query[i][j].resize(o + 1);
            value1[i][j].resize(o + 1);
            value2[i][j].resize(o + 1);
        }
    }
    for (int i = 1; i <= m; ++i)
    {
        query[i][0][0] = X.substr(0, i);
        value1[i][0][0] = value1[i - 1][0][0] + '-';
        value2[i][0][0] = value2[i - 1][0][0] + '-';
    }
    for (int i = 1; i <= n; ++i)
    {
        query[0][i][0] = query[0][i - 1][0] + '-';
        value1[0][i][0] = Y.substr(0, i);
        value2[0][i][0] = value2[0][i][0] + '-';
    }
    for (int i = 1; i <= o; ++i)
    {
        query[0][0][i] = query[0][0][i - 1] + '-';
        value1[0][0][i] = value1[0][0][i - 1] + '-';
        value2[0][0][i] = Z.substr(0, i);
    }
    
    // pre-compute 3 surfaces
    vector<int> cand;
    for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= n; ++j)
        {
            char c1 = X[i - 1], c2 = Y[j - 1];
            cand.clear();
            cand.push_back(dp[i - 1][j - 1][0] + alpha(c1, c2) + 2 * delta);
            cand.push_back(dp[i - 1][j][0] + 2 * delta);
            cand.push_back(dp[i][j - 1][0] + 2 * delta);
            dp[i][j][0] = *min_element(cand.begin(), cand.end());
            // store route
            if (cand[0] == dp[i][j][0])
            {
                query[i][j][0] = query[i - 1][j - 1][0] + c1;
                value1[i][j][0] = value1[i - 1][j - 1][0] + c2;
                value2[i][j][0] = value2[i - 1][j - 1][0] + '-';
            }
            else if (cand[1] == dp[i][j][0])
            {
                query[i][j][0] = query[i - 1][j][0] + c1;
                value1[i][j][0] = value1[i - 1][j][0] + '-';
                value2[i][j][0] = value2[i - 1][j][0] + '-';
            }
            else if (cand[2] == dp[i][j][0])
            {
                query[i][j][0] = query[i][j - 1][0] + '-';
                value1[i][j][0] = value1[i][j - 1][0] + c2;
                value2[i][j][0] = value2[i][j - 1][0] + '-';
            }
            else
                cout << "ERROR\n";
        }
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= o; ++j)
        {
            char c1 = Y[i - 1], c2 = Z[j - 1];
            cand.clear();
            cand.push_back(dp[0][i - 1][j - 1] + alpha(c1, c2) + 2 * delta);
            cand.push_back(dp[0][i - 1][j] + 2 * delta);
            cand.push_back(dp[0][i][j - 1] + 2 * delta);
            dp[0][i][j] = *min_element(cand.begin(), cand.end());
            if (cand[0] == dp[0][i][j])
            {
                query[0][i][j] = query[0][i - 1][j - 1] + '-';
                value1[0][i][j] = value1[0][i - 1][j - 1] + c1;
                value2[0][i][j] = value2[0][i - 1][j - 1] + c2;
            }
            else if (cand[1] == dp[0][i][j])
            {
                query[0][i][j] = query[0][i - 1][j] + '-';
                value1[0][i][j] = value1[0][i - 1][j] + c1;
                value2[0][i][j] = value2[0][i - 1][j] + '-';
            }
            else if (cand[2] == dp[0][i][j])
            {
                query[0][i][j] = query[0][i][j - 1] + '-';
                value1[0][i][j] = value1[0][i][j - 1] + '-';
                value2[0][i][j] = value2[0][i][j - 1] + c2;
            }
            else
                cout << "ERROR\n";
        }
    for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= o; ++j)
        {
            char c1 = X[i - 1], c2 = Z[j - 1];
            cand.clear();
            cand.push_back(dp[i - 1][0][j - 1] + alpha(c1, c2) + 2 * delta);
            cand.push_back(dp[i - 1][0][j] + 2 * delta);
            cand.push_back(dp[i][0][j - 1] + 2 * delta);
            dp[i][0][j] = *min_element(cand.begin(), cand.end());
            if (cand[0] == dp[i][0][j])
            {
                query[i][0][j] = query[i - 1][0][j - 1] + c1;
                value1[i][0][j] = value1[i - 1][0][j - 1] + '-';
                value2[i][0][j] = value2[i - 1][0][j - 1] + c2;
            }
            else if (cand[1] == dp[i][0][j])
            {
                query[i][0][j] = query[i - 1][0][j] + c1;
                value1[i][0][j] = value1[i - 1][0][j] + '-';
                value2[i][0][j] = value2[i - 1][0][j] + '-';
            }
            else if (cand[2] == dp[i][0][j])
            {
                query[i][0][j] = query[i][0][j - 1] + '-';
                value1[i][0][j] = value1[i][0][j - 1] + '-';
                value2[i][0][j] = value2[i][0][j - 1] + c2;
            }
            else
                cout << "ERROR\n";
        }
    // compute in cubic
    for (int i = 1; i <= m; ++i)
        for (int j = 1; j <= n; ++j)
            for (int k = 1; k <= o; ++k)
            {
                char c1 = X[i - 1], c2 = Y[j - 1], c3 = Z[k - 1];
                cand.clear();

                cand.push_back(dp[i - 1][j - 1][k - 1] + alpha(c1, c2) + alpha(c2, c3) + alpha(c1, c3));

                cand.push_back(dp[i - 1][j - 1][k] + alpha(c1, c2) + 2 * delta);
                cand.push_back(dp[i - 1][j][k - 1] + alpha(c1, c3) + 2 * delta);
                cand.push_back(dp[i][j - 1][k - 1] + alpha(c2, c3) + 2 * delta);

                cand.push_back(dp[i][j][k - 1] + 2 * delta);
                cand.push_back(dp[i][j - 1][k] + 2 * delta);
                cand.push_back(dp[i - 1][j][k] + 2 * delta);
                dp[i][j][k] = *min_element(cand.begin(), cand.end());
                if (cand[0] == dp[i][j][k])
                {
                    query[i][j][k] = query[i - 1][j - 1][k - 1] + c1;
                    value1[i][j][k] = value1[i - 1][j - 1][k - 1] + c2;
                    value2[i][j][k] = value2[i - 1][j - 1][k - 1] + c3;
                }
                else if (cand[1] == dp[i][j][k])
                {
                    query[i][j][k] = query[i - 1][j - 1][k] + c1;
                    value1[i][j][k] = value1[i - 1][j - 1][k] + c2;
                    value2[i][j][k] = value2[i - 1][j - 1][k] + '-';
                }
                else if (cand[2] == dp[i][j][k])
                {
                    query[i][j][k] = query[i - 1][j][k - 1] + c1;
                    value1[i][j][k] = value1[i - 1][j][k - 1] + '-';
                    value2[i][j][k] = value2[i - 1][j][k - 1] + c3;
                }
                else if (cand[3] == dp[i][j][k])
                {
                    query[i][j][k] = query[i][j - 1][k - 1] + '-';
                    value1[i][j][k] = value1[i][j - 1][k - 1] + c2;
                    value2[i][j][k] = value2[i][j - 1][k - 1] + c3;
                }
                else if (cand[4] == dp[i][j][k])
                {
                    query[i][j][k] = query[i][j][k - 1] + '-';
                    value1[i][j][k] = value1[i][j][k - 1] + '-';
                    value2[i][j][k] = value2[i][j][k - 1] + c3;
                }
                else if (cand[5] == dp[i][j][k])
                {
                    query[i][j][k] = query[i][j - 1][k] + '-';
                    value1[i][j][k] = value1[i][j - 1][k] + c2;
                    value2[i][j][k] = value2[i][j - 1][k] + '-';
                }
                else if (cand[6] == dp[i][j][k])
                {
                    query[i][j][k] = query[i - 1][j][k] + c1;
                    value1[i][j][k] = value1[i - 1][j][k] + '-';
                    value2[i][j][k] = value2[i - 1][j][k] + '-';
                }
                else
                    cout << "ERROR\n";
            }
    res r;
    r.cost = dp[m][n][o];
    r.q = query[m][n][o];
    r.v1 = value1[m][n][o];
    r.v2 = value2[m][n][o];
    return r;
}

int main()
{
    clock_t timer_start = clock();
    // read data
    ifstream query_in("./query.txt", ifstream::in);
    // ifstream db_in("../data/MSA_database.txt", ifstream::in);
    ifstream db_in("../data/toy_database.txt", ifstream::in);
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
    // Search
    res final_res;
    for (int i = 0; i < db.size(); ++i)
    {
        cout<<i<<endl;
        for (int j = i + 1; j < db.size(); ++j)
        {
            string seq1 = db[i];
            string seq2 = db[j];
            res cur_res = DP(query, seq1, seq2, 2);
            final_res = cur_res.cost < final_res.cost ? cur_res : final_res;
        }
    }
    printf("min_cost: %d \n", final_res.cost);
    cout << final_res.q << endl;
    cout << final_res.v1 << endl;
    cout << final_res.v2 << endl;
    clock_t timer_end = clock();
    cout << "Running time: " << (timer_end - timer_start) * 1000 / CLOCKS_PER_SEC << "ms" << endl;
}