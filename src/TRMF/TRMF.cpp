#include <../include/TRMF/TRMF.h>

using arma::mat;
using arma::vec;

TRMF::TRMF(mat &data, mat &idmat, arma::uvec const &lags, size_t rank, arma::vec const &lambda, double eta,
           size_t maxiter) : m_data(data),
                             m_id(idmat),
                             m_lags(lags),
                             m_lambdas(lambda),
                             m_rank(rank),
                             m_eta(eta),
                             m_maxiter(maxiter)
{

    m_nb_lags = m_lags.n_rows;

    //Initialisation
    m_W = mat(m_data.n_rows, m_rank, arma::fill::ones);
    m_X = mat(m_data.n_cols, m_rank, arma::fill::ones);
    m_T = mat(m_nb_lags, m_rank, arma::fill::ones);

    m_binary = mat(m_id.n_rows, m_id.n_cols, arma::fill::zeros);
    m_binary.elem(arma::find(m_id > 0)).ones();

    m_Mt = mat(m_rank, m_rank, arma::fill::zeros);
    m_Nt = vec(m_rank, arma::fill::zeros);

    m_Pt = mat(m_rank, m_rank, arma::fill::zeros);
    m_Qt = arma::rowvec(m_rank, arma::fill::zeros);

    m_rank_eye = mat(m_rank, m_rank, arma::fill::eye);
}

void TRMF::fit()
{
    for (size_t it = 0; it < m_maxiter; ++it)
    {
        mat var1 = m_X.t();
        mat var2 = kr_prod(var1, var1);
        mat var3 = var2 * m_binary.t();
        mat var4 = var1 * m_id.t();

        arma::uvec index;
        mat theta0;

        for (int i = 0; i < m_id.n_rows; ++i)
        {
            mat v3 = var3.col(i);
            v3.reshape(m_rank, m_rank);
            m_W.row(i) = ((v3 + m_lambdas(0) * m_rank_eye).i() * var4.col(i)).t();
        }

        var1 = m_W.t();
        var2 = kr_prod(var1, var1);
        var3 = var2 * m_binary;
        var4 = var1 * m_id;
        
        for (size_t t = 0; t < m_id.n_cols; ++t)
        {
            m_Mt.zeros();
            m_Nt.zeros();

            if (t < max(m_lags))
            {
                m_Pt.zeros();
                m_Qt.zeros();
            }
            else
            {
                m_Pt.eye();
                m_Qt = sum(m_T % m_X.rows(t - m_lags), 0);
            }
            if (t < m_id.n_cols - min(m_lags))
            {
                if (t >= max(m_lags) and t < m_id.n_cols - max(m_lags))
                {
                    index = arma::linspace<arma::uvec>(0, m_nb_lags - 1, m_nb_lags);
                }
                else
                {
                    index = arma::find(((t + m_lags) >= arma::max(m_lags)) and (t + m_lags) < m_id.n_cols);
                }
                for (const auto &k : index)
                {
                    
                     theta0 = m_T;
                     theta0.row(k).zeros();                   
                     m_Mt = m_Mt + arma::diagmat(arma::square(m_T.row(k)));
                }
                
            }
            else
            {
            }
        }
    }
}

arma::mat TRMF::kr_prod(arma::mat const &A, arma::mat const &B)
{
    arma::mat result = arma::mat(A.n_rows * B.n_rows, B.n_cols, arma::fill::zeros);
    for (int i = 0; i < A.n_cols; ++i)
    {
        result.col(i) = arma::kron(A.col(i), B.col(i));
    }
    return result;
}