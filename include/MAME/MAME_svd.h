//
// Created by daniel on 05.11.19.
//

#ifndef MAME_MAME_SVD_H
#define MAME_MAME_SVD_H

#include <armadillo>

class MAME_svd {

public:

    MAME_svd(arma::vec& data, size_t N, int sv);

    void fit();
    double predict(arma::vec oldata);

    arma::vec &m_data;
    int m_sv;
    size_t m_n;
    size_t m_m;

    arma::mat m_M;
    arma::mat m_nM;
    arma::mat m_W;

    arma::mat m_U;
    arma::mat m_V;
    arma::vec m_S;

    arma::mat m_nU;
    arma::mat m_nV;
    arma::vec m_nS;




};


#endif //MAME_MAME_SVD_H
