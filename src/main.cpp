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


std::string input = "../IO/datasets/electricity.txt";
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

static void TEST_LSRN(int ts)
{

    size_t STEPS = 100;

    //Load data
    arma::mat data;
    data.load(input);
    double min = data.min();

    arma::vec vadd(STEPS, arma::fill::zeros);
    arma::rowvec radd(STEPS + 1, arma::fill::zeros);

    arma::mat concat;
    arma::mat W;
    W.load("/home/daniel/Desktop/mpelec/W/W" + std::to_string(ts + 1) + ".txt");
    concat = arma::join_rows(W, vadd);
    W = arma::join_cols(concat, radd);

    //Algorithm doesn't handle negative values.
    arma::mat train = data.rows(0, 5000 - (STEPS + 1));
    train.transform([min](double val) { return (val + (-1) * min); });

    arma::mat real = data.rows(5000 - (STEPS), 4999);

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

    predV.save(output + "predV" + std::to_string(ts + 1) + ".txt", arma::raw_ascii);
    realV.save(output + "realV" + std::to_string(ts + 1) + ".txt", arma::raw_ascii);

    double rmse = sqrt(arma::mean(arma::pow(realV - predV, 2)));
    std::cout << ts << " :: RMSE = " << rmse << std::endl;
}

static void TEST_MAME()
{

    arma::vec test(90, arma::fill::ones);
    for (size_t i = 0; i < 90; ++i)
    {
        test(i) = i;
    }
    MAME_svd mame = MAME_svd(test, 5, 0);

    //  mame.m_M.print();
    std::cout << "-------------" << std::endl;

    mame.fit();
    std::cout << "pred = " << mame.predict(test.rows(86, 89)) << std::endl;
}

static void TEST_TRMF()
{
    int STEPS = 10;

    arma::uvec time_lags;
    arma::vec lambdas;
    arma::mat data;

    double eta = 0.1;
    size_t maxiter = 100;

    time_lags << 1 << 2;
    lambdas << 0.75 << 0.75 << 0.75;

    data = arma::mat(5, 100, arma::fill::ones);
    data = data.cols(0, data.n_cols - (STEPS + 1));
    std::cout << "data = (" << data.n_rows << "," << data.n_cols << ")" << std::endl;
    std::cout << data << std::endl;

    TRMF trmf = TRMF(data, data, time_lags, 5, lambdas, eta, maxiter);
    trmf.fit();

    arma::mat X = trmf.m_X;
    arma::mat W = trmf.m_W;
    arma::mat Theta = trmf.m_T;

    std::cout << "X = (" << X.n_rows << "," << X.n_cols << ")" << std::endl;
    std::cout << X << std::endl;
    std::cout << "W = (" << W.n_rows << "," << W.n_cols << ")" << std::endl;
    std::cout << W << std::endl;
    std::cout << "Theta = (" << Theta.n_rows << "," << Theta.n_cols << ")" << std::endl;
    std::cout << Theta << std::endl;
    std::cout << "result" << std::endl;
    arma::mat temp = (W * X.t());
    std::cout << "temp = (" << temp.n_rows << "," << temp.n_cols << ")" << std::endl;

    std::cout << temp.cols(data.n_cols - (STEPS + 1), (data.n_cols) - 1) << std::endl;
}

static void TEST_TRMF2()
{
    int STEPS = 10;

    arma::uvec time_lags;
    arma::vec lambdas;
    arma::mat data;

    double eta = 0.09;
    size_t maxiter = 500;

    time_lags << 1 << 2;
    lambdas << 0.75 << 0.75 << 0.75;

    data.load(input);
    data = data.t();
    
    arma::mat pred = TRMF::one_pred(data, data, time_lags, 10, lambdas, eta, maxiter, STEPS, 20);
    std::cout << "pred = (" << pred.n_rows << "," << pred.n_cols << ")" << std::endl;
    std::cout << pred << std::endl;
}

static void TEST_TRMF3()
{

    size_t STEPS = 50;
    size_t MULTI_STEPS = 5;

    arma::uvec time_lags;
    arma::vec lambdas;
  

    double eta = 0.1;
    size_t maxiter = 600;

    time_lags << 1 << 2 << 3 << 4 << 5 << 6 << 7 << 8 << 9 << 10;
    lambdas << 0.75 << 0.75 << 0.75;
    size_t RANK = 21;

    arma::mat data;
    data.load(input);
    data = data.t();
    // data = data.rows(0,1);
    // data = data.cols(0,100 - 1);

    std::cout << "data = (" << data.n_rows << "," << data.n_cols << ")" << std::endl;
    
    arma::mat pred = TRMF::multi_pred(data, data, time_lags, RANK, lambdas, eta, maxiter, STEPS, MULTI_STEPS);
    std::cout << "pred = (" << pred.n_rows << "," << pred.n_cols << ")" << std::endl;

    pred.save(output + "TRMF/50forecast5by5cpp" + ".txt", arma::raw_ascii);

}

int main()
{
    //ss

    //  arma::vec time_lags;
    //  time_lags << 1 << 2;

    //  std::cout<< 22 - time_lags <<std::endl;

    //  uvec index = linspace<uvec>(0, 4-1, 4);
    //  std::cout<< index <<std::endl;

    TEST_TRMF3();

    return 0;
}