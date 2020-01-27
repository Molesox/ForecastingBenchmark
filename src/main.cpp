#include <iostream>
#include <armadillo>
#include <vector>
#include <MAME/MAME_svd.h>
#include <TRMF/TRMF.h>

#include "OMF/TemplateOMF.h"
#include "OTS/TemplateOTS.h"
#include "OATS/TemplateOATS.h"

#include "OATS/OATS_ogd.h"

#include "OTS/OTS_ogd.h"
#include "OTS/OTS_gsr.h"

#include "OMF/FixedPenalty.h"
#include "OMF/FixedTolerance.h"
#include "OMF/ZeroTolerance.h"

#include "LSRN/LatentSpaceRN.h"

#include <chrono>

std::string input = "../IO/datasets/trmfSynt.txt";
std::string output = "../IO/outputs/";

static void TEST_OMF()
{

    arma::mat data;
    data.load(input);
    data = arma::trans(data);

    TemplateOMF *omf[] = {
        new FixedTolerance(data),
        new FixedPenalty(data),
        new ZeroTolerance(data)};

    for (auto &algo : omf)
    {
        std::cout << "Algo : " << algo->m_name << " is forecasting." << std::endl;

        arma::mat pred = algo->forecast();
        pred = arma::trans(pred);
        pred.save(output + "OMF/" + algo->m_name + "c++.txt", arma::raw_ascii);
        delete algo;
    }
}

static void TEST_OTS()
{

    size_t STEPS = 10;

    arma::mat data;
    data.load(input);

    for (size_t i = (5000 - STEPS - 1); i < 5000; ++i)
    {
        for (size_t j = 0; j < 20; ++j)
        {
            data(i, j) = NAN;
        }
    }

    TemplateOTS *ots[]{
        new OTS_ogd(data),
        new OTS_gsr(data)};

    for (auto &algo : ots)
    {
        std::cout << "Algo : " << algo->m_name << " is forecasting." << std::endl;
        arma::mat pred = algo->forecast();
        pred = pred.t();
        pred.save(output + "OTS/" + algo->m_name + ".txt", arma::raw_ascii);
        delete algo;
    }
}

static void TEST_OATS()
{
    arma::mat data;
    data.load(input);

    TemplateOATS *oats[]{
        new OATS_ogd(data)};

    for (auto &algo : oats)
    {
        std::cout << "Algo : " << algo->m_name << " is forecasting." << std::endl;
        arma::mat pred = algo->forecast();
        pred.save(output + "OATS/" + algo->m_name + ".txt", arma::raw_ascii);
        delete algo;
    }
}

static void TEST_LSRN()
{
    input = "../IO/datasets/LSMRN/";
    size_t STEPS = 100;
    size_t ts = 0;
    //Load data
    arma::mat data;
    data.load(input + "traffic.txt");
    std::cout << "data = (" << data.n_rows << "," << data.n_cols << ")" << std::endl;

    double min = data.min();
    std::cout << min << std::endl;

    arma::vec vadd(STEPS, arma::fill::zeros);
    arma::rowvec radd(STEPS + 1, arma::fill::zeros);

    //arma::mat concat;
    arma::mat W;
    W.load(input + "adj_mat.txt");
    std::cout << "W = (" << W.n_rows << "," << W.n_cols << ")" << std::endl;

    //concat = arma::join_rows(W, vadd);
    //W = arma::join_cols(concat, radd);

    //Algorithm doesn't handle negative values.
    arma::mat train = data.rows(0, 5000 - (STEPS + 1));
    train.transform([min](double val) { return (val + (-1) * min); });

    arma::mat real = data.rows(5000 - (STEPS), 5000 - 1);

    arma::vec realV = real.col(ts);
    arma::vec topred = train.col(ts);

    arma::mat pred;
    LatentSpaceRN lsrn = LatentSpaceRN(topred, STEPS, 0.0002, 1e-5, W, 20, 2);
    lsrn.do_globalLearning();
    pred = lsrn.forecast();

    //Unshift values
    pred.transform([min](double val) { return (val + min); });
    arma::vec predV(pred.n_rows - 1);
    for (size_t i = 0; i < pred.n_rows - 1; i++)
    {
        predV(i) = pred(i, i + 1);
    }

    predV.save(output + "LSMRN/predV" + std::to_string(ts + 1) + ".txt", arma::raw_ascii);
    realV.save(output + "LSMRN/realV" + std::to_string(ts + 1) + ".txt", arma::raw_ascii);

    double rmse = sqrt(arma::mean(arma::pow(realV - predV, 2)));
    std::cout << ts << " :: RMSE = " << rmse << std::endl;
}

static void TEST_MAME()
{
    input = "/mnt/c/Users/Daniel/Desktop/FBthesis/mame/combinedTS.txt";
    output = "../IO/outputs/MAME/";
    int N = 50;
    int M = 400;
    size_t PRED = 2000;

    arma::vec data;
    arma::vec train;
    arma::vec test;

    data.load(input);
    train = data.rows(0, data.n_rows - PRED - 1);
    test = data.rows(data.n_rows - PRED, data.n_rows - 1);

    std::cout << "train = (" << train.n_rows << "," << train.n_cols << ")" << std::endl;
    std::cout << "data = (" << data.n_rows << "," << data.n_cols << ")" << std::endl;

    MAME_svd mame = MAME_svd(train, N, 2);
    mame.fit();

    arma::vec pred(PRED, arma::fill::zeros);
    arma::vec pastvals(N - 1, arma::fill::zeros);
    size_t j;
    for (size_t i = 0; i < PRED; i++)
    {
        pastvals.zeros();
        j = 0;

        if (i < N - 1)
        {
            while (j < N - 1 - i)
            {
                pastvals(j) = data(train.n_rows - (N - 1 - i) + j);
                j += 1;
            }
        }
        if (j < N - 1)
        {
            size_t calc = (i - (i - (size_t)(N - 1) + j));
            pastvals.rows(j, j + calc - 1) = test.rows(i - (size_t)(N - 1) + j, i - 1);
        }
        pred(i) = mame.predict(pastvals);
    }
    pred.save(output + "mamecpp.txt", arma::raw_ascii);
}

static void TEST_TRMF()
{
    arma::mat Y(100, 20, arma::fill::ones);
    arma::mat Yt(20, 100, arma::fill::ones);
    arma::uvec lagset(5, arma::fill::ones);
    int i = 0;
    for (auto &el : lagset)
    {
        el = i + 1;
        i++;
    }
    size_t k = 5;
    trmf_prob_t prob(Y, Yt, lagset, k);
    trmf_param_t param = trmf_param_t();

    arma::mat W, H, lag_val;
    trmf_initialization(prob, param, W, H, lag_val);
    check_dimension(prob, param, W, H, lag_val);

    trmf_train(prob, param, W, H, lag_val);
}

static void TEST_TRMF_ROLLING()
{
    std::string input = "../IO/datasets/electricity_normal.txt";
    output = "../IO/outputs/TRMF/";

    arma::mat data;
    data.load(input);

    trmf_param_t param = trmf_param_t();
    param.lambdaI = 0.5;
    param.lambdaAR = 125;
    param.lambdaLag = 2;
    size_t RANK = 30;

    arma::uvec lagset;
    lagset << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9 << 10 << 11 << 12 << 13 << 14 << 15 << 16 << 17 << 18 << 19 << 20 << 21 << 22 << 23 << 24 << 168 << 169 << 170 << 171 << 172 << 173 << 174 << 175 << 176 << 177 << 178 << 179 << 180 << 181 << 182 << 183 << 184 << 185 << 186 << 187 << 188 << 189 << 190 << 191;

    arma::mat pred;
    pred = multi_pred(data, param, 24, 7,lagset,RANK);

    pred.save(output + "newCppp.txt", arma::raw_ascii);
}

int main()
{

    TEST_TRMF_ROLLING();
    //ss

    //  arma::vec time_lags;
    //  time_lags << 1 << 2;

    //  std::cout<< 22 - time_lags <<std::endl;

    //  uvec index = linspace<uvec>(0, 4-1, 4);
    //  std::cout<< index <<std::endl;

    //    TEST_TRMF();

    return 0;
}