#include "OATS/OATS_ogd.h"

OATS_ogd::OATS_ogd(arma::mat &data, int m, int k, double eta)
    : TemplateOATS(data, m, k, eta)
{
    m_lambda = arma::rowvec(m + k, arma::fill::randu); //lambda_{j} € [0,1] for all j.
    m_name = "OATS_ogd";
}

arma::vec OATS_ogd::vec_forecast(arma::vec const &x)
{
    size_t T = x.n_rows;
    double diff = 0.;
    arma::vec x_pred = arma::vec(T, arma::fill::zeros);
    m_lambda.randu();

    int mk = m_m + m_k;

    for (size_t t = mk + 1; t < T; t++)
    {
        //prediction step:
        x_pred(t) = arma::as_scalar(m_lambda * x.rows(t - mk, t - 1));

        //by now only first order difference.
        diff = x_pred(t) - x(t);

        //update step:
        m_lambda = m_lambda - (x.rows(t - mk, t - 1).t() * 2 * diff) / sqrt(t - mk) * m_eta;
    }

    return x_pred;
}