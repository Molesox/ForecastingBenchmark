#include "BEATLEX/Beatlex.h"
#include <cmath>
#include <algorithm>
using arma::mat;
using arma::span;
using arma::uvec;
using arma::vec;
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

    model_momentum = 0.8;
    max_vocab = 5;
    totalerr = 0;
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
    arma::uword ave_min = ave_cost.index_min();
    arma::uvec min = arma::ind2sub(arma::size(ave_cost), ave_min).head_rows(2);

    return std::make_tuple(min(0), min(1));
}

void Beatlex::summarize_seq()
{
    auto best_initial_tup = new_segment(1);
    size_t best_initial = std::get<0>(best_initial_tup);

    ends.push_back(best_initial);
    starts.push_back(1);
    idx.push_back(1);
    models.push_back(X.cols(starts.front(), ends.front()));

    double clust_threshold = 0.3;
    double mean_dev = arma::mean(arma::square(X.as_col() - arma::mean(X.as_col())));
    double best_pref_lenght = NAN;
    size_t Xcols = X.n_cols;
    bool done = false;
    while (ends.back() < Xcols and not done)
    {
        size_t curr_idx = starts.size() + 1;
        size_t curr = ends.back();
        size_t num_models = models.size();
        starts.push_back(curr);
        mat ave_cost(num_models, Smax);
        ave_cost.fill(arma::datum::inf);

        size_t cur_end = std::min(curr + Smax - 1, Xcols);
        mat Xcur = X.cols(curr, cur_end - 1);
        dtw_t res;
        mat dtwcost;

        for (size_t k = 0; k < num_models; k++)
        {
            res = dtw(models[k], Xcur, maxdist);
            ave_cost(arma::span(k), arma::span(0, Xcur.n_cols - 1)) = res.D.row(res.D.n_rows - 1) / arma::regspace(1, Xcur.n_cols).t();
            ave_cost(arma::span(k), arma::span(0, Smin - 1)).fill(arma::datum::nan);
        }

        auto best_idx = ave_cost.index_min();
        auto best_cost = ave_cost(best_idx);

        uvec min = arma::ind2sub(arma::size(ave_cost), best_idx).head_rows(2);
        size_t best_k = min(0);
        size_t best_size = min(1);

        vec good_prefix_costs;
        vec good_prefix_length;
        mat ave_prefix_cost;
        if (curr + Smax > Xcols)
        {
            good_prefix_costs = vec(num_models).fill(arma::datum::nan);
            good_prefix_length = vec(num_models).fill(arma::datum::nan);
            dtw_t res;
            for (size_t k = 0; k < num_models; k++)
            {
                res = dtw(models[k], Xcur, maxdist);
                ave_prefix_cost = res.D.col(res.D.n_cols - 1) / arma::regspace(1, models[k].n_cols);

                auto best_prefix_idx = ave_prefix_cost.index_min();
                auto best_prefix_cost = ave_prefix_cost(best_prefix_idx);
                good_prefix_costs(k) = best_prefix_cost;
                good_prefix_length(k) = best_prefix_idx;
            }

            auto best_prefix_k = good_prefix_costs.index_min();
            auto best_prefix_cost = good_prefix_costs(best_prefix_k);
            auto best_prefix_length = good_prefix_length(best_prefix_k);

            if (best_prefix_cost < best_cost)
            {
                ends[curr_idx] = Xcols;
                idx[curr_idx] = best_prefix_k;
                break;
            }
        }
        mat Xbest = X.cols(curr, curr + best_size - 1);
        if (best_cost > clust_threshold * mean_dev and models.size() < max_vocab)
        {
            auto best_S1 = std::get<0>(new_segment(curr));
            ends[curr_idx] = curr + best_S1 - 1;
            idx[curr_idx] = num_models + 1;
            models.push_back(X.cols(starts[curr_idx], ends[curr_idx]));
            totalerr += clust_threshold * mean_dev * best_S1;
        }
        else
        {
            ends[curr_idx] = curr + best_size - 1;
            idx[curr_idx] = best_k;
            totalerr += best_cost * best_size;
            dtw_t res = dtw(models[best_k], Xbest, maxdist);
            mat trace_sum(arma::size(models[best_k]), arma::fill::zeros);
            for (size_t t = 0; t < res.w.n_cols; t++)
            {
                trace_sum.col(res.w(t, 0)) = trace_sum.col(res.w(t, 0)) + Xbest.col(res.w(t, 1));
            }
            vec b = arma::unique(res.w.col(0));
            uvec c = arma::hist(res.w.col(0), b);
            arma::urowvec trace_counts = arma::conv_to<arma::urowvec>::from(c);
            models[best_k] = model_momentum * models[best_k] + (1. - model_momentum) * trace_sum / c;
        }

        /*

        trace_ave = bsxfun(@rdivide, trace_summed, trace_counts);
        models{best_k} = model_momentum * models{best_k} + (1 - model_momentum) * trace_ave;
        
        */

        done = true;
    }
}