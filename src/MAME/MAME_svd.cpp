//
// Created by daniel on 05.11.19.
//

#include "MAME/MAME_svd.h"

MAME_svd::MAME_svd(arma::vec &data, size_t N, int sv)
        : m_data(data), m_sv(sv), m_n(N), m_m(data.n_rows / N)
{
    m_M = arma::reshape(m_data, m_n, m_m);
}

void MAME_svd::fit()
{

    arma::svd(m_U, m_S, m_V, m_M);

    m_S = m_S.rows(0, m_sv);
    m_U = m_U.cols(0, m_sv);
    m_V = m_V.cols(0, m_sv);

    m_M = m_U * arma::diagmat(m_S) * m_V.t();

    m_nM = arma::mat(m_n - 1, m_m, arma::fill::zeros);

    int row = 0;
    int mat = 0;
    int tsrow = m_n - 1;

    while (row < m_n - 1)
    {
        std::cout << "m_nM = (" << m_nM.n_rows << "," << m_nM.n_cols << ")" << std::endl;
        std::cout << "m_M = (" << m_M.n_rows << "," << m_M.n_cols << ")" << std::endl;
        std::cout << "row = " << row << std::endl;
        std::cout << "mat = " << mat << std::endl;
        std::cout << "each = " << tsrow << std::endl;

        m_nM.rows(row, row + (tsrow-1))  = m_M.rows(mat, mat + (tsrow-1));
        std::cout << "m_nM = (" << m_nM.n_rows << "," << m_nM.n_cols << ")" << std::endl;

        row += tsrow;
        mat += m_n;

    }

    arma::svd(m_nU, m_nS, m_nV, m_nM);

    m_nS = m_nS.rows(0, m_sv);
    m_nU = m_nU.cols(0, m_sv);
    m_nV = m_nV.cols(0, m_sv);

    m_nS.for_each([](arma::mat::elem_type &val){ if (val > 0) val = 1. / val; });

    m_W = (m_nU * arma::diagmat(m_nS) * m_nV.t()) * m_M.row(m_n - 1).t();


}

double MAME_svd::predict(arma::vec oldata)
{
    std::cout << "W = \n"<< m_W << std::endl;
    std::cout << "old data = \n"<<oldata << std::endl;
    return arma::as_scalar(m_W.t() * oldata);
}

