#include "OTS/OTS_gsr.h"

OTS_gsr::OTS_gsr(arma::mat &data, size_t d, double eta)
    : TemplateOTS(data, eta), m_lag(d)
{
    m_name = "ots_gsr";
}

arma::mat OTS_gsr::inner_product(arma::vec const &y, size_t d)
{
    size_t n = y.n_rows;
    arma::mat prod = arma::mat(n, n, arma::fill::zeros);

    for (size_t t = 2; t <= n; t++)
    {
        for (size_t s = 2; s <= t; s++)
        {
            size_t counter = 0;
            for (size_t k = 1; k <= std::min(d, s - 1); k++)
            {
                if (not std::isnan(y((t - 1) - k)) && not std::isnan(y((s - 1) - k)))
                {
                    prod((t - 1), (s - 1)) += y((t - 1) - k) * y((s - 1) - k) * pow(2, counter);
                }
                if (std::isnan(y((t - 1) - k)) && std::isnan(y((s - 1) - k)))
                {
                    counter += 1;
                }
            }
        }
    }
    return prod;
}
arma::vec OTS_gsr::vec_forecast(arma::vec const &x)
{
    size_t n = x.n_rows;
    double eta = m_eta / sqrt(n);

    arma::vec err = arma::vec(n, arma::fill::zeros);
    arma::vec x_pred = arma::vec(n, arma::fill::zeros);
    arma::mat prod = inner_product(x, m_lag);

    for (size_t t = 1; t < n; t++)
    {

        arma::colvec error = (arma::colvec)(err.subvec(0, t - 1));
        arma::vec numerator = prod(arma::span(t, t), arma::span(0, t - 1)) * error;
        arma::vec denominator = error.t() * prod(arma::span(0, t - 1), arma::span(0, t - 1)) * error;

        //TODO: change to as_scalar().
        x_pred(t) = (eta * numerator(0)) / (fmax(1., eta * sqrt(denominator(0))));

        if (not std::isnan(x(t)))
        {
            err(t) = x(t) - x_pred(t);
        }
    }

    return x_pred;
}