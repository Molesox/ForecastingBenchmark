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
    m_W = 0.1 * mat(m_data.n_rows, m_rank, arma::fill::randn);  //random
    m_X = 0.1 * mat(m_data.n_cols, m_rank, arma::fill::randn);  //random
    m_T = 0.1 * mat(m_nb_lags, m_rank, arma::fill::randn);      //random

    m_Mt = mat(m_rank, m_rank, arma::fill::zeros);
    m_Nt = vec(m_rank, arma::fill::zeros);

    m_Pt = mat(m_rank, m_rank, arma::fill::zeros);
    m_Qt = arma::rowvec(m_rank, arma::fill::zeros);

    m_rank_eye = mat(m_rank, m_rank, arma::fill::eye);
}

void TRMF::fit()
{
    m_binary = mat(m_id.n_rows, m_id.n_cols, arma::fill::zeros);
    m_binary.elem(arma::find(m_id != 0)).ones();
    for (size_t it = 0; it < m_maxiter; ++it)
    {
        mat var1 = m_X.t();
        mat var2 = kr_prod(var1, var1);
        mat var3 = var2 * m_binary.t();
        mat var4 = var1 * m_id.t();

        arma::uvec index;
        mat theta0;
        mat inv;

        for (int i = 0; i < m_id.n_rows; ++i)
        {
            mat v3 = var3.col(i);
            v3.reshape(m_rank, m_rank);
            m_W.row(i) = (arma::inv(v3 + m_lambdas(0) * m_rank_eye) * var4.col(i)).t();
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
                m_Qt = arma::sum(m_T % m_X.rows(t - m_lags), 0);
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
                    mat temp = (m_X.row(t + m_lags(k)) - arma::sum(theta0 % m_X.rows(t + m_lags(k) - m_lags)));
                    m_Nt = m_Nt + (m_T.row(k) % temp).t();
                }
         
                
                inv = arma::inv(arma::reshape(var3.col(t), m_rank, m_rank) + m_lambdas(1) * (m_Pt + m_Mt + m_eta * m_rank_eye));
                m_X.row(t) = (inv * (var4.col(t) + m_lambdas(1) * (m_Qt.t() + m_Nt))).t();
            }
            else
            {
                
                inv = arma::inv(arma::reshape(var3.col(t), m_rank, m_rank) + m_lambdas(1) * (m_Pt + m_eta * m_rank_eye));
                m_X.row(t) = (inv * (var4.col(t) + m_Qt.t())).t();
            }
        }
        for (size_t k = 0; k < m_nb_lags; k++)
        {
            var1 = m_X.rows(arma::max(m_lags) - m_lags(k), m_id.n_cols - m_lags(k) - 1);
            var2 = arma::inv(arma::diagmat(arma::sum(var1 % var1)) + (m_lambdas(2) / m_lambdas(1)) * m_rank_eye);
            var3 = vec(m_rank, arma::fill::zeros);

            for (size_t t = (arma::max(m_lags) - m_lags(k)); t < m_id.n_cols - m_lags(k); t++)
            {
                var3 = var3 + (m_X.row(t) % (m_X.row(t + m_lags(k)) - arma::sum(m_T % m_X.rows(t + m_lags(k) - m_lags)) + (m_T.row(k) % m_X.row(t)))).t();
            }
            m_T.row(k) = (var2 * var3).t();
        }

    }
}

arma::mat TRMF::one_pred(mat &data, mat &idmat, arma::uvec const &lags, size_t rank, vec const &lambdas, double eta, size_t maxiter, size_t pred_steps, size_t back_steps)
{
    size_t start_time = data.n_cols - pred_steps;

    mat data0 = data.cols(0, start_time - 1);
    mat id0 = idmat.cols(0, start_time - 1);

    size_t dim1 = id0.n_rows;
    size_t dim2 = id0.n_cols;

    TRMF trmf(data0, id0, lags, rank, lambdas, eta, maxiter);
    trmf.fit();

    mat X0 = mat(dim2 + 1, rank, arma::fill::zeros);
    X0.rows(0, dim2 - 1) = trmf.m_X.rows(0, dim2 - 1);
    X0.row(dim2) = arma::sum(trmf.m_T % X0.rows(dim2 - lags));

    trmf.m_X = X0.rows(X0.n_rows - back_steps, X0.n_rows - 1);

    mat pred_mat = mat(dim1, pred_steps, arma::fill::zeros);
    pred_mat.col(0) = trmf.m_W * X0.row(dim2).t();

    trmf.m_maxiter = 100;
    for (size_t t = 1; t < pred_steps; t++)
    {
        trmf.m_data = data.cols(start_time - back_steps + t, start_time + t - 1);
        trmf.m_id = idmat.cols(start_time - back_steps + t, start_time + t - 1);

        trmf.fit();

        X0.zeros(back_steps + 1, rank);
        X0.rows(0, back_steps - 1) = trmf.m_X.rows(0, back_steps - 1);
        X0.row(back_steps) = arma::sum(trmf.m_T % X0.rows(back_steps - lags));

        trmf.m_X = X0.rows(1, back_steps);
        pred_mat.col(t) = trmf.m_W * X0.row(back_steps).t();
    }
    return pred_mat;
}

arma::mat TRMF::multi_pred(mat &data, mat &idmat, arma::uvec const &lags, size_t rank,
                           vec &lambdas, double eta, size_t maxiter, size_t pred_steps, size_t multi_steps)
{
    size_t start_time = data.n_cols - pred_steps;
    
    
    mat data0 = data.cols(0, start_time - 1);
    mat id0 = idmat.cols(0, start_time - 1);   
    
    size_t dim1 = id0.n_rows;
    size_t dim2 = id0.n_cols;

    mat pred(dim1, pred_steps, arma::fill::zeros);

    TRMF trmf(data0, id0, lags, rank, lambdas, eta, maxiter);
    trmf.fit();      

    mat X0 = mat(dim2 + multi_steps, rank, arma::fill::zeros);
    X0.rows(0, dim2 - 1) = trmf.m_X.rows(0, dim2 - 1);

    for (size_t t0 = 0; t0 < multi_steps; t0++)
    {
        X0.row(dim2 + t0) = arma::sum(trmf.m_T % X0.rows(dim2 + t0 - lags));
    }

    trmf.m_X = X0;
    mat temp = trmf.m_W * X0.rows(dim2, dim2 + multi_steps - 1).t();
    pred.cols(0, multi_steps - 1) = temp.cols(temp.n_cols - multi_steps, temp.n_cols - 1);
    std::cout<< "main fit done." <<std::endl;
    
    trmf.m_maxiter = 200;
    for (size_t t = 1; t < (size_t)(pred_steps / multi_steps); t++)
    {
        trmf.m_data = data.cols(0, start_time + t * multi_steps - 1);
        trmf.m_id = idmat.cols(0, start_time + t * multi_steps - 1);
        trmf.fit();

        dim1 = trmf.m_id.n_rows;
        dim2 = trmf.m_id.n_cols;
        X0.zeros(dim2 + multi_steps, rank);
        X0.rows(0, dim2 - 1) = trmf.m_X.rows(0, dim2 - 1);

        for (size_t t0 = 0; t0 < multi_steps; t0++)
        {
            X0.row(dim2 + t0) = arma::sum(trmf.m_T % X0.rows(dim2 + t0 - lags));
        }
        trmf.m_X = X0;
        pred.cols(t * multi_steps, (t + 1) * multi_steps - 1) = trmf.m_W * X0.rows(dim2, dim2 + multi_steps - 1).t();
        std::cout<< "." <<std::flush;
    }
    std::cout<<std::endl;
    
    return pred;
}
arma::mat TRMF::kr_prod(arma::mat const &A, arma::mat const &B)
{
    mat result = mat(A.n_rows * B.n_rows, B.n_cols, arma::fill::zeros);
    for (int i = 0; i < A.n_cols; ++i)
    {
        result.col(i) = arma::kron(A.col(i), B.col(i));
    }
    return result;
}