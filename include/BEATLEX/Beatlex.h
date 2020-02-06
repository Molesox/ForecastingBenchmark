#ifndef BEATLEX_BEATLEX_H
#define BEATLEX_BEATLEX_H

#include <armadillo>
struct dtw_t
{
    arma::mat D; //accumulated distance matrix
    arma::mat w; //optimal path
    double k;    //normalizing factor
    double Dist; //unnormalized distance between t and r

    void dtw(arma::mat t, arma::mat r, double max_dis);
};
arma::mat distance(const arma::mat &X, const arma::mat &Y);
class Beatlex
{
private:
    /*  */
public:
    Beatlex();
    ~Beatlex();
};

#endif //BEATLEX_BEATLEX_H