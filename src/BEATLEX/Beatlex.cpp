#include "BEATLEX/Beatlex.h"
using arma::mat;
arma::mat distance(const arma::mat &X, const arma::mat &Y)
{
    arma::mat ones(1, X.n_cols, arma::fill::ones);
    arma::mat D = arma::sum(arma::square(X), 1) * ones + ones.t() * arma::sum(arma::square(Y), 1).t() - 2 * X * Y.t();
    
    return D;
}

void dtw_t::dtw(mat t, mat r, double max_dis){

    size_t rows = t.n_rows;
    size_t N = t.n_cols;
    size_t M = r.n_cols;

}