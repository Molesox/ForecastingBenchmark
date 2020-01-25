#include <armadillo>
#ifndef VERBOSE
#define VERBOSE false
#endif

void log1(std::string text);
class function
{

public:
    virtual double fun(arma::mat &w) = 0;
    virtual void grad(arma::mat &w, arma::mat &g) = 0;
    virtual void Hv(arma::mat &s, arma::mat &Hs) = 0;
    virtual ~function() {}
    virtual void init() {}
};
class tron
{
private:
    double m_eps;
    double m_eps_cg;
    size_t m_maxiter;
    size_t m_maxiter_cg;
    size_t m_n;
    function &m_fun;

    arma::mat s;
    arma::mat r;
    arma::mat w_new;
    arma::mat g;
    //.......
    arma::mat d;
    arma::mat Hd;

public:
    tron(function &fun, double eps, double eps_cg, size_t maxiter, size_t maxiter_cg);
    ~tron();
    void trust_region(arma::mat &w, bool set2zero = true);
    void set_eps(double eps, double eps_cg = 0.1)
    {
        m_eps = eps;
        m_eps_cg = eps_cg;
    }
    void set_solver(arma::mat& w, bool set2zero)
    {
        log1("TRON:: setting solver");
        trust_region(w, set2zero);
        log1("TRON:: solver set");
    }

private:
    size_t trcg(double delta, arma::mat &g, arma::mat &s, arma::mat &r, double cg_rnorm);
};
