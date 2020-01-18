
#include <armadillo>
#include <vector>

class NormalTransform
{

private:
    arma::mat m_mean;
    arma::mat m_std;

    arma::mat m_a;
    arma::mat m_b;

public:
    NormalTransform(arma::mat &Y);
    arma::mat preprocess(arma::mat &Y);
    arma::mat posprocess(arma::mat &Y);
};

class TRMF
{

public:
    explicit TRMF(arma::mat &data, arma::mat &idmat, arma::uvec const &lags, size_t rank, arma::vec const &lambda,
                  double eta, size_t maxiter);
    void fit();
    static arma::mat one_pred(arma::mat &data, arma::mat &idmat, arma::uvec const &lags, size_t rank, arma::vec const &lambdas, double eta, size_t maxiter, size_t pred_steps, size_t back_steps);
    static arma::mat multi_pred(arma::mat &data, arma::mat &idmat, arma::uvec const &lags, size_t rank, arma::vec &lambdas, double eta, size_t maxiter, size_t pred_steps, size_t multi_steps);
    //protected:

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

    arma::mat m_temp;
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