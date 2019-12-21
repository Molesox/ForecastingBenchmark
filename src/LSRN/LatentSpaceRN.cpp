#include "LSRN/LatentSpaceRN.h"

LatentSpaceRN::LatentSpaceRN(arma::vec &data, size_t steps, double gamma, double lambda, arma::mat Wprox, size_t d,
                             size_t khop)
        : m_data(data), STEPS(steps), m_gamma(gamma), m_lambda(lambda), m_W(Wprox), m_khop(khop)
{

    m_d = d;
    m_B = arma::mat(m_d, m_d, arma::fill::randu);
    m_A = arma::mat(m_d, m_d, arma::fill::randu);


    m_iter = data.n_rows / STEPS;
    for (size_t i = 0; i < m_iter; i++)
    {
        mv_G.emplace_back(STEPS + 1, STEPS + 1, arma::fill::zeros);
        mv_Y.emplace_back(STEPS + 1, STEPS + 1, arma::fill::zeros);
        mv_U.emplace_back(STEPS + 1, m_d, arma::fill::randu);

        for (size_t col = 0; col < mv_G[i].n_rows - 1; col++)
        {
            mv_Y[i](col, col + 1) = 1;
            mv_G[i](col, col + 1) = m_data(col + (i * STEPS));
        }
    }

    W();
    m_D = arma::mat(STEPS + 1, STEPS + 1, arma::fill::zeros);
    
    for (size_t i = 0; i < m_W.n_rows; i++)
    {
        for (size_t j = 0; j < m_W.n_cols; j++)
        {
            m_D(i, i) += m_W(i, j);
        }
    }


}

void LatentSpaceRN::do_globalLearning()
{

    arma::mat nomi;
    arma::mat nomiB(m_d, m_d, arma::fill::zeros);
    arma::mat denoB(m_d, m_d, arma::fill::zeros);
    arma::mat nomiA(m_d, m_d, arma::fill::zeros);
    arma::mat denoA(m_d, m_d, arma::fill::zeros);
    int maxIter = 0;
    do
    {
        //std::cout << ".";

        for (size_t t = 0; t < m_iter; t++)
        {
            //update U

            if (t == 0)
            {
                nomi = (mv_Y[t] % mv_G[t]) * (mv_U[t] * m_B.t() + mv_U[t] * m_B) + m_lambda * m_W * mv_U[t] +
                       m_gamma * (mv_U[t + 1] * m_A.t());
            } else if (t == m_iter - 1)
            {
                nomi = (mv_Y[t] % mv_G[t]) * (mv_U[t] * m_B.t() + mv_U[t] * m_B) + m_lambda * m_W * mv_U[t] +
                       m_gamma * (mv_U[t - 1] * m_A);
            } else
            {
                nomi = (mv_Y[t] % mv_G[t]) * (mv_U[t] * m_B.t() + mv_U[t] * m_B) + m_lambda * m_W * mv_U[t] +
                       m_gamma * (mv_U[t - 1] * m_A + mv_U[t + 1] * m_A.t());
            }

            arma::mat deno = (mv_Y[t] % (mv_U[t] * m_B * mv_U[t].t())) * (mv_U[t] * m_B.t() + mv_U[t] * m_B) +
                             m_lambda * m_D * mv_U[t] + m_gamma * (mv_U[t] + mv_U[t] * m_A * m_A.t());

            mv_U[t] = mv_U[t] % arma::pow((nomi / deno), 1. / 4);
        }

        for (size_t t = 0; t < m_iter; t++)
        {

            //update B
            nomiB = nomiB + mv_U[t].t() * (mv_Y[t] % mv_G[t]) * mv_U[t];
            denoB = denoB + mv_U[t].t() * (mv_Y[t] % (mv_U[t] * m_B * mv_U[t].t())) * mv_U[t];

            //update A
            if (t > 0)
            {
                nomiA = nomiA + mv_U[t - 1].t() * mv_U[t];
                denoA = denoA + mv_U[t - 1].t() * mv_U[t - 1] * m_A;
            }
        }
        m_oA = m_A;
        m_oB = m_B;

        m_B = m_B % (nomiB / denoB);
        m_A = m_A % (nomiA / denoA);
        maxIter++;
    } while (not converge() and maxIter < 200);
    if (maxIter > 198)
    {
        std::cerr << "max iter = " << maxIter << std::endl;
    }
}

arma::mat LatentSpaceRN::forecast()
{
    arma::mat pred = (mv_U.back() * m_A) * m_B * (mv_U.back() * m_A).t();
    return pred;
}

LatentSpaceRN::~LatentSpaceRN()
{
}

bool LatentSpaceRN::converge()
{
    return (arma::norm(m_A - m_oA, "fro") < 1e-4) && (arma::norm(m_B - m_oB, "fro") < 1e-4);
}

void LatentSpaceRN::W()
{
    m_W = arma::mat(STEPS + 1, STEPS + 1, arma::fill::zeros);
/*
    arma::vec vadd(STEPS, arma::fill::zeros);
    arma::rowvec radd(STEPS + 1, arma::fill::zeros);

    arma::mat concat;
    arma::mat temp;

    temp.load("/home/daniel/CLionProjects/LSMRN/IO/dataset/profMatElec.txt");
    concat = arma::join_rows(temp, vadd);
    temp = arma::join_cols(concat, radd);
    m_W = temp;

    for (int i = 1; i < 51; ++i) {
        temp.load("/home/daniel/Desktop/mpelec/proxi/W" + std::to_string(i) + ".txt");
        concat = arma::join_rows(temp, vadd);
        temp = arma::join_cols(concat, radd);

        mv_W.push_back(temp);

}*/

    for (size_t i = 1; i <= m_khop; ++i)
    {
        m_W += armaPow(mv_Y[0], i);

    }
}

arma::mat LatentSpaceRN::armaPow(arma::mat const &toP, size_t n)
{

    arma::mat result = toP;

    if (n == 1)
    {
        return toP;
    }
    for (size_t i = 1; i < n; ++i)
    {

        result *= toP;

    }

    return result;
}
