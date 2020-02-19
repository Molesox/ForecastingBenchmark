#ifndef BEATLEX_BEATLEX_H
#define BEATLEX_BEATLEX_H

#include <armadillo>
#include <tuple>
#include <vector>
#include <list>

struct dtw_t
{
    arma::mat D; //accumulated distance matrix
    arma::mat w; //optimal path
    double k;    //normalizing factor
    double Dist; //unnormalized distance between t and r
};

dtw_t dtw(arma::mat t, arma::mat r, size_t max_dis);
arma::mat distance(const arma::mat &X, const arma::mat &Y);


class Beatlex
{
private:

    const arma::mat& X;
    arma::mat Xp;
    std::vector<arma::mat> models;

    std::vector<size_t> starts;
    std::vector<size_t> ends;
    std::vector<size_t> idx;
    
    size_t Smin;
    size_t Smax;
    size_t maxdist;
    size_t pred_steps;
    

    double totalerr;
    double model_momentum;
    size_t max_vocab;


public:
    Beatlex(arma::mat& data, size_t smin, size_t smax, size_t maxdist, size_t predsteps);
    std::tuple<size_t,size_t> new_segment(size_t cur);
    void summarize_seq();
   
    
};

#endif //BEATLEX_BEATLEX_H