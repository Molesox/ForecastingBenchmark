
#include <armadillo>
#include <vector>

class TRMF
{

public:


    explicit TRMF(arma::mat &data, arma::mat &idmat, arma::uvec const &lags, size_t rank, arma::vec const &lambda,
                  double eta, size_t maxiter);
    void fit();


protected:

    arma::mat &m_data;
    arma::mat &m_id;

    arma::mat m_W;
    arma::mat m_X;
    arma::mat m_T;

    arma::uvec m_lags;
    arma::vec m_lambdas;

    arma::mat m_binary;
    arma::mat m_rank_eye;

    arma::mat m_Mt;
    arma::vec m_Nt;

    arma::mat m_Pt;
    arma::rowvec m_Qt;
/*
    size_t m_id_nrows;
    size_t m_id_ncols;

    size_t m_dat_nrows;
    size_t m_dat_ncols;
*/
    size_t m_rank;
    size_t m_nb_lags;
    size_t m_maxiter;

    double m_eta;

private:

    arma::mat kr_prod(arma::mat const &A, arma::mat const &B);


};