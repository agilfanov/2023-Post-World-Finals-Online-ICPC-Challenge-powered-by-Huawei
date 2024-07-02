#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <queue>
#include <stack>
#include <cmath>
#include <random>
#include <chrono>



using namespace std;

typedef long long ll;

#define endl "\n"

const ll mod = 1;
const ll INF = (ll)1e18;

random_device rd;
mt19937 gen(rd());

class Float16{
    static const uint32_t mantissaShift = 42;
    static const uint32_t expShiftMid   = 56;
    static const uint32_t expShiftOut   = 52;
    double dValue_;

public:
    Float16(double in) : dValue_(in) {
        uint64_t utmp;
        memcpy(&utmp, &dValue_, sizeof utmp);
        utmp = utmp >> mantissaShift;
        utmp = utmp << mantissaShift;
        const uint64_t maskExpMid = (63llu << expShiftMid);
        const uint64_t maskExpOut = (15llu << expShiftOut);
        const uint64_t maskExpLead = (1llu << 62);
        const uint64_t maskMantissaD = (1llu << 63) + maskExpLead + maskExpMid + maskExpOut;
        if (utmp & maskExpLead) {
            if (utmp & maskExpMid) {
                utmp = utmp | maskExpMid;
                utmp = utmp & maskMantissaD;
                utmp = utmp | maskExpOut;
            }
        } else {
            if ((utmp & maskExpMid) != maskExpMid) {
                utmp = 0;
            }
        }
        memcpy(&dValue_, &utmp, sizeof utmp);
    }

    Float16() : dValue_(0) {}

    Float16& operator=(const Float16& rhs) {
        this->dValue_ = rhs.dValue_;
        return *this;
    }

    Float16& operator=(const double& rhs) {
        this->dValue_ = rhs;
        uint64_t utmp;
        memcpy(&utmp, &dValue_, sizeof utmp);
        utmp = utmp >> mantissaShift;
        utmp = utmp << mantissaShift;
        memcpy(&dValue_, &utmp, sizeof utmp);
        return *this;
    }

    friend Float16 operator+(const Float16& lhs, const Float16& rhs) {
        double tmp = lhs.dValue_ + rhs.dValue_;
        return Float16(tmp);
    }

    double convert2Double() { return dValue_; }
};

double add64(double a, double b) {
    volatile double currResultDouble = a + b;
    return currResultDouble;
}

double add32(double a, double b) {
    float currResultSingle = static_cast<float>(a) + static_cast<float>(b);

    return static_cast<double>(currResultSingle);
}

double add16(double a, double b) {
    Float16 currResultHalf(0.);
    currResultHalf = currResultHalf + Float16(a);
    currResultHalf = currResultHalf + Float16(b);
    return currResultHalf.convert2Double();
}

double true_sum(const vector<double>& vec) {
    long double trueSum = 0;
    long double correction = 0;
    vector<double> dvtmp=vec;
    sort(dvtmp.begin(),dvtmp.end(), [](const double x, const double y) {
        return fabs(x) < fabs(y);
    });
    for (auto i : dvtmp) {
        volatile long double y = static_cast<long double>(i) - correction;
        volatile long double t = trueSum + y;
        correction = (t - trueSum) - y;
        trueSum = t;
    }
    return (double)trueSum;
}


double error(double a, double b) {

    double s = a + b;
    double a_ = s - b;
    double b_ = s - a_;
    double da = a - a_;
    double db = b - b_;
    double t = da + db;
    return t;
}

int base10power(double x) {
    if (x == 0) return 0;
    return (int)floor(log10(abs(x)));

}

vector<pair<double, string>> input;


bool cmp(pair<double, string>& x, pair<double, string>& y) {
    return abs(x.first) < abs(y.first);
}

struct group {
    int l, r;
    double avg;

    group(int a, int b, double c) {
        l = a;
        r = b;
        avg = c;
    }
};


bool cmpGroup(group a, group b) {
    return abs(a.avg) < abs(b.avg);
}



struct state {
    double sum;
    int weight;
    string answer;
};


pair<double, string> divideAndConquer(int left, int right) {

    if (left == right) {
        if (left >= input.size() || left < 0) {
            cout << "s";
        }
        return {input[left].first, input[left].second};
    }
    std::uniform_int_distribution<> random(left, right - 1);

    int m = random(gen);


    auto l = divideAndConquer(left, m);
    auto r = divideAndConquer(m + 1, right);

    long double sum = l.first + r.first;
    double sum32 = (double)((float)l.first) + (double)((float)r.first);
    double sum16 = add16(l.first, r.first);
    pair<double, string> ret;
    if (sum16 == sum) {
        ret.first = sum16;
        ret.second = "{h:" + l.second + "," + r.second + "}";
    } else if (sum32 == sum) {
        ret.first = sum32;
        ret.second = "{s:" + l.second + "," + r.second + "}";
    } else {
        ret.first = ((double)l.first + (double)r.first);
        ret.second = "{d:" + l.second + "," + r.second + "}";
    }
    return ret;
}


pair<double, string> divideAndConquerEven(int left, int right) {

    if (left == right) {
        return {input[left].first, input[left].second};
    }
    int m = (left + right) / 2;

    auto l = divideAndConquerEven(left, m);
    auto r = divideAndConquerEven(m + 1, right);

    long double sum = l.first + r.first;
    double sum32 = (double)((float)l.first) + (double)((float)r.first);
    double sum16 = add16(l.first, r.first);
    pair<double, string> ret;
    if (sum16 == sum) {
        ret.first = sum16;
        ret.second = "{h:" + l.second + "," + r.second + "}";
    } else if (sum32 == sum) {
        ret.first = sum32;
        ret.second = "{s:" + l.second + "," + r.second + "}";
    } else {
        ret.first = ((double)l.first + (double)r.first);
        ret.second = "{d:" + l.second + "," + r.second + "}";
    }
    return ret;
}

void updateAns(double real, pair<double, string>& best, pair<double, string>& curr) {
    long double diff = abs((long double)real - best.first);
    long double curr_diff = abs((long double)real - curr.first);
    if (curr_diff < diff) {
        best = curr;
    }
}

void solve() {

    auto beg = chrono::high_resolution_clock::now();



    int n;
    cin >> n;
    input.resize(n);
    vector<double> allNums (n);
    for (int i = 0; i < n; i++) {
        cin >> input[i].first;
        input[i].second = to_string(i + 1);
        allNums[i] = input[i].first;
    }


    vector<group> groups;

    for (int i = 0; i < n; i += 16) {
        sort(input.begin() + i, min(input.end(), input.begin() + i + 16), cmp);
        long double sum = 0;
        vector<double> curr;
        for (int j = i; j < min(n, i + 16); j++) {
            curr.push_back(input[j].first);
        }

        sum = true_sum(curr);
        if (i + 16 <= n) {
            groups.emplace_back(i, min(n - 1, i + 15), sum);
        }

    }
    sort(groups.begin(), groups.end(), cmpGroup);

    vector<pair<double, string>> copy = input;


    for (int i = 0; i < n; i += 16) {
        if (i + 15 < n) {
            for (int j = i; j < min(n, i + 16); j++) {
                input[j] = copy[groups[i / 16].l + j - i];
            }
        }
        if (i > 0) {
            int lastExp = base10power(input[i - 1].first);

            if (abs(lastExp - base10power(input[i].first)) > abs(lastExp - base10power(i))) {
                reverse(input.begin() + i, min(input.end(), input.begin() + i + 16));
            }
        }

    }



    pair<double, string> answer;
    double real_sum = true_sum(allNums);
    answer = divideAndConquerEven(0, n - 1);



    if (n <= 900) {
        vector<vector<double>> true_sums (n, vector<double> (n));
        vector<vector<state>> dp (n, vector<state> (n));

        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                vector<double> nums;
                for (int k = i; k <= j; k++) nums.push_back(input[k].first);
                true_sums[i][j] = true_sum(nums);
            }
        }

        for (int i = 0; i < n; i++) {
            dp[i][i].sum = input[i].first;
            dp[i][i].weight = 0;
            dp[i][i].answer = input[i].second;
        }

        for (int sz = 2; sz <= n; sz++) {
            for (int l = 0; l + sz - 1 < n; l++) {
                int r = l + sz - 1;
                long double smallest_error = 1e306;
                double best_sum;
                int smallest_weight = 1e9;
                int smallest_m;
                int type_did;
                for (int m = l; m < r; m++) {
                    long double current_sum16 = abs(true_sums[l][r] - add16(dp[l][m].sum, dp[m + 1][r].sum));
                    long double current_sum32 = abs(true_sums[l][r] - add32(dp[l][m].sum, dp[m + 1][r].sum));
                    long double current_sum64 = abs(true_sums[l][r] - add64(dp[l][m].sum, dp[m + 1][r].sum));
                    int weight_before = dp[l][m].weight + dp[m + 1][r].weight;

                    if (current_sum16 < smallest_error || (current_sum16 == smallest_error && smallest_weight > weight_before + 1)) {
                        smallest_m = m;
                        smallest_error = current_sum16;
                        smallest_weight = weight_before + 1;
                        best_sum = add16(dp[l][m].sum, dp[m + 1][r].sum);
                        type_did = 16;
                    }
                    if (current_sum32 < smallest_error || (current_sum32 == smallest_error && smallest_weight > weight_before + 2)) {
                        smallest_m = m;
                        smallest_error = current_sum32;
                        smallest_weight = weight_before + 2;
                        best_sum = add32(dp[l][m].sum, dp[m + 1][r].sum);
                        type_did = 32;

                    }
                    if (current_sum64 < smallest_error || (current_sum64 == smallest_error && smallest_weight > weight_before + 4)) {
                        smallest_m = m;
                        smallest_error = current_sum64;
                        smallest_weight = weight_before + 4;
                        best_sum = add64(dp[l][m].sum, dp[m + 1][r].sum);
                        type_did = 64;
                    }

                }
                dp[l][r].sum = best_sum;
                dp[l][r].weight = smallest_weight;
                if (type_did == 16) {
                    dp[l][r].answer = "{h:" + dp[l][smallest_m].answer + "," + dp[smallest_m + 1][r].answer + "}";
                } else if (type_did == 32) {
                    dp[l][r].answer = "{s:" + dp[l][smallest_m].answer + "," + dp[smallest_m + 1][r].answer + "}";
                } else {
                    dp[l][r].answer = "{d:" + dp[l][smallest_m].answer + "," + dp[smallest_m + 1][r].answer + "}";
                }
            }
        }
        pair<double, string> dpAns = make_pair(dp[0][n - 1].sum, dp[0][n - 1].answer);
        updateAns(real_sum, answer, dpAns);

    }

    int curr = 0;
    vector<pair<double, string>> useGreedy = input;
    int cycles = 0;
    while (useGreedy.size() != 1) {
        cycles++;
        vector<pair<double, string>> next;
        for (int l = 0; l < useGreedy.size(); l++) {
            long double trueSum = useGreedy[l].first, correction = 0;

            double sum = useGreedy[l].first;
            deque<string> current;
            current.emplace_back(useGreedy[l].second);
            int r {};
            for (r = l + 1; r < useGreedy.size(); r++) {
                if (abs(base10power(sum) - base10power(useGreedy[r].first)) > curr) break;
                volatile long double y = static_cast<long double>(useGreedy[r].first) - correction;
                volatile long double t = trueSum + y;
                correction = (t - trueSum) - y;
                trueSum = t;

                long double sum16 = abs(trueSum - (long double)add16(sum, useGreedy[r].first));
                long double sum32 = abs(trueSum - (long double)add32(sum, useGreedy[r].first));
                long double sum64 = abs(trueSum - (long double)add64 (sum, useGreedy[r].first));
                long double best = min({sum16, sum32, sum64});
                if (sum16 <= best) {
                    sum = add16(sum, useGreedy[r].first);
                    current.emplace_front("{h:");
                    current.emplace_back("," + useGreedy[r].second + "}");
                } else if (sum32 <= best) {
                    sum = add32(sum, useGreedy[r].first);
                    current.emplace_front("{s:");
                    current.emplace_back("," + useGreedy[r].second + "}");
                } else {
                    sum = add64 (sum, useGreedy[r].first);
                    current.emplace_front("{d:");
                    current.emplace_back("," + useGreedy[r].second + "}");
                }
            }
            l = r - 1;
            string ret;
            int sz = (int)current.size();
            for (int i = 0; i < sz; i++) {
                ret += current.front();
                current.pop_front();
            }
            next.emplace_back(sum, ret);
        }

        useGreedy = next;
        curr++;
    }
    pair<double, string> greedyAns = make_pair(useGreedy[0].first, useGreedy[0].second);
    updateAns(real_sum, answer, greedyAns);


    int times = 0;
    long double nlog = n * log2(n);
    for (times = 0; times * nlog <= 3e8; times++);

    for (int i = 0; i < times; i++) {
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds >(end - beg);
        if (duration.count() + (int)(nlog / (2e8)) >= 8400) break;
        pair<double, string> current_roll = divideAndConquer(0, n - 1);
        updateAns(real_sum, answer, current_roll);
    }

    cout << answer.second;

}

int main() {

    //   freopen("input.txt", "r", stdin);
    //   freopen("output.txt", "w", stdout);
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);


    solve();


    return 0;
}