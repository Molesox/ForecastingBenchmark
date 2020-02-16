#include "BEATLEX/Beatlex.h"
#include <cmath>
#include <algorithm>
using arma::mat;
using arma::span;
arma::mat distance(const arma::mat &X, const arma::mat &Y)
{
    arma::mat XX, YY, XY, D;
    arma::rowvec rones(Y.n_cols, arma::fill::ones);
    arma::colvec cones(X.n_cols, arma::fill::ones);

    XX = arma::sum(X % X, 0);
    YY = arma::sum(Y % Y, 0);
    XY = X.t() * Y;
    D = XX.t() * rones + cones * YY - 2 * XY;

    return D;
}

dtw_t dtw(mat t, mat r, size_t max_dis)
{
    dtw_t ret;

    size_t rows = t.n_rows;
    size_t N = t.n_cols;
    size_t M = r.n_cols;

    mat d = distance(t, r) / rows;

    ret.D.set_size(arma::size(d));
    ret.D.fill(arma::datum::inf);
    ret.D(0, 0) = d(0, 0);

    for (size_t n = 1; n < N; n++)
    {
        ret.D(n, 0) = d(n, 0) + ret.D(n - 1, 0);
    }
    for (size_t m = 1; m < M; m++)
    {
        ret.D(0, m) = d(0, m) + ret.D(0, m - 1);
    }

    double mcost = arma::stddev(arma::vectorise(t)) * log2((double)M);
    double ncost = arma::stddev(arma::vectorise(r)) * log2((double)N);
    size_t m_min, m_max;

    for (size_t n = 2; n <= N; n++)
    {
        m_min = std::max(2, (int)n - (int)max_dis);
        m_max = std::min(M, n + (size_t)max_dis);

        for (size_t m = m_min; m <= m_max; m++)
        {
            ret.D(n - 1, m - 1) = d(n - 1, m - 1) + std::min({ret.D(n - 2, m - 1) + mcost, ret.D(n - 2, m - 2), ret.D(n - 1, m - 2) + ncost});
        }
    }

    ret.Dist = ret.D(N - 1, M - 1);
    ret.k = 1;
    ret.w << N << M;

    arma::rowvec tmp;

    while (N + M != 2)
    {
        if (N - 1 == 0)
        {
            M = M - 1;
        }
        else if (M - 1 == 0)
        {
            N = N - 1;
        }
        else
        {
            auto mmin = std::min({ret.D(N - 2, M - 1), ret.D(N - 1, M - 2), ret.D(N - 2, M - 2)});

            if (mmin == ret.D(N - 2, M - 1))
            {
                N--;
            }
            else if (mmin == ret.D(N - 1, M - 2))
            {
                M--;
            }
            else
            {
                N--;
                M--;
            }
        }
        ret.k++;
        tmp << N << M;
        ret.w = arma::join_vert(ret.w, tmp);
    }
    return ret;
}

Beatlex::Beatlex(mat &data, size_t smin, size_t smax, size_t maxdist,
                 size_t predsteps) : X(data), Smin(smin), Smax(smax), maxdist(maxdist)
{
}

std::tuple<size_t, size_t> Beatlex::new_segment(size_t cur)
{
    size_t num_models = models.size();

    arma::cube ave_cost(Smax, num_models + 1, Smax);
    ave_cost.fill(arma::datum::inf);

    arma::mat current_model;
    arma::mat Xcur;
    dtw_t dtw_res;

    for (size_t S = Smin; S <= Smax; S++)
    {
        if (cur + S >= X.n_cols)
        {
            continue;
        }

        for (size_t k = 1; k <= num_models + 1; k++)
        {
            if (k <= num_models)
            {
                current_model = models[k];
            }
            else
            {
                current_model = X.cols(cur - 1, cur + S - 2);
            }
            size_t min = std::min((size_t)X.n_cols, cur + S + Smax - 1);

            Xcur = X.cols(cur + S - 1, min - 1);
            dtw_res = dtw(current_model, Xcur, maxdist);

            ave_cost(span(S - 1), span(k - 1), span(0, Xcur.n_cols - 1)) = dtw_res.D.row(dtw_res.D.n_rows - 1) / arma::regspace(1, Xcur.n_cols).t();
            ave_cost(span(S - 1), span(k - 1), span(0, Smin - 1)).fill(arma::datum::inf);
        }
    }
    arma::uword a = ave_cost.index_min();
    arma::vec min = arma::min(ave_cost);

    return std::make_tuple(a, 0);
}
/*
[~, best_idx] = nanmin(ave_costs(:));
[best_S1, best_k, ~] = ind2sub(size(ave_costs), best_idx);
disp([best_S1, best_k])
*/