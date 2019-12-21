//
// Created by daniel on 15.10.19.
//

#ifndef LSMRN_LATENTSPACERN_H
#define LSMRN_LATENTSPACERN_H


#include <armadillo>
#include <iostream>
#include <vector>

class LatentSpaceRN
{
public:
    LatentSpaceRN(arma::vec &data, size_t steps, double gamma, double lambda, arma::mat W, size_t d = 20,
                  size_t k_hop = 2);

    void do_globalLearning();

    arma::mat forecast();

    ~LatentSpaceRN();

protected:
    arma::vec &m_data;


    size_t m_d;//dimension of latent space
    size_t STEPS;
    size_t m_khop;
    size_t m_iter;
    double m_gamma;
    double m_lambda;

    std::vector<arma::mat> mv_G;    //adj matrices.
    std::vector<arma::mat> mv_Y;    //indication matrix Yi,j = 1 <==> Gi,j != 0
    std::vector<arma::mat> mv_U;
    // std::vector<arma::mat> mv_W;
    // std::vector<arma::mat> mv_D;

    arma::mat m_B; //latent space matrix.
    arma::mat m_A; //transistion matrix
    arma::mat m_W; //proximity matrix
    arma::mat m_D;

    arma::mat m_oB; //old latent space matrix.
    arma::mat m_oA; //old transistion matrix

private:
    bool converge();

    void W();

    arma::mat armaPow(arma::mat const &toP, size_t n);
};


#endif //LSMRN_LATENTSPACERN_H
