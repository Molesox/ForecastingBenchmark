
#include <armadillo>
#include <assert.h>
#include <omp.h>
#include "tron.h"
void logger(std::string text);
struct arr_prob_t
{
    arma::mat &Y;        // m*n sparse matrix
    arma::mat &H;        // n*k array row major H(j,t) = H[k*j+t]
    arma::uvec *lag_set; // lag set L (sorted is needed)
    arma::mat *lag_val;  // |L| * k , i.e. Theta  COLMAJOR

    size_t m; // #time stamps = Y.rows
    size_t n; // #time series = Y.cols
    size_t k; // #low-rank dimension

    arr_prob_t(arma::mat &Y, arma::mat &H, arma::uvec *lag_set, arma::mat *lag_val) : Y(Y),
                                                                                      H(H),
                                                                                      n(Y.n_cols),
                                                                                      m(Y.n_rows),
                                                                                      k(H.n_cols),
                                                                                      lag_set(lag_set),
                                                                                      lag_val(lag_val)
    {
        if (lag_set != NULL)
        {
            assert(lag_set->n_elem == lag_val->n_rows);
            assert(lag_val->n_cols == H.n_cols);
        }
    }
};

struct arr_param_t
{

    double lambdaAR;
    double lambdaI;
    double eps;    // eps for TRON
    double eps_cg; // eps used for CG

    size_t max_tron_iter;
    size_t max_cg_iter;

    arr_param_t()
    {
        lambdaI = 0.1;
        lambdaAR = 0.1;
        eps = 0.1;
        eps_cg = 0.1;
        max_tron_iter = 2;
        max_cg_iter = 10;
    }
};

struct trmf_prob_t
{
    arma::mat &Y;        // m*n sparse matrix
    arma::mat &Yt;       // Y transpose
    arma::uvec &lag_set; // lag set L (sorted is needed)
    size_t m;            // #time stamps = Y.rows
    size_t n;            // #time series = Y.cols
    size_t k;            // #low-rank dimension

    trmf_prob_t(arma::mat &Y, arma::mat &Yt, arma::uvec &lag_set, size_t k) : Y(Y),
                                                                              Yt(Yt),
                                                                              lag_set(lag_set),
                                                                              m(Y.n_rows),
                                                                              n(Y.n_cols),
                                                                              k(k) {}
};

struct trmf_param_t : public arr_param_t
{
    double lambdaLag;
    size_t max_iter;
    int period_W;
    int period_H;
    int period_Lag;

    trmf_param_t() : arr_param_t()
    {
        lambdaLag = 0.1;
        max_iter = 40;
        period_W = 1;
        period_H = 1;
        period_Lag = 2;
    }
};

struct solver_t
{
    virtual void init_prob() = 0;
    virtual void solve(arma::mat &W) = 0;
    virtual double solver_fun(arma::mat &W) { return 0.; }
    virtual ~solver_t() {}
};

struct arr_solver : public solver_t
{ // {{{
    arr_prob_t *prob;
    arr_param_t *param;
    function *fun_obj;
    tron *tron_obj;
    solver_t *solver_obj;
    bool done_init;

    arr_solver(arr_prob_t *prob, arr_param_t *param);
    arr_solver(const arr_solver &other) {}

    void zero_init()
    {
        prob = NULL;
        param = NULL;
        fun_obj = NULL;
        tron_obj = NULL;
        solver_obj = NULL;
        done_init = false;
    }

    ~arr_solver()
    {
        if (tron_obj)
        {
            delete tron_obj;
        }
        if (fun_obj)
        {
            delete fun_obj;
        }
        if (solver_obj)
        {
            delete solver_obj;
        }
        zero_init();
    }

    void init_prob()
    {

        if (fun_obj)
        {
            logger("arr_solver::init_prob() calling fun_obj->init()");

            fun_obj->init();
        }
        else if (solver_obj)
        {
            logger("arr_solver::init_prob() calling solver_obj->init_prob()");
            solver_obj->init_prob();
        }
        done_init = true;
    }

    void set_eps(double eps) { tron_obj->set_eps(eps); }

    void solve(arma::mat &w)
    {
        logger("arr_solver::solve()");

        if (!done_init)
        {
            logger("arr_solver::init prob");
            this->init_prob();
            logger("arr_solver::init prob done.");
        }
        if (tron_obj)
        {
            logger("arr_solver::setting tron");
            bool set_w_to_zero = false;
            w = w.as_col();
            tron_obj->set_solver(w, set_w_to_zero);
            w.reshape(prob->m, prob->k);
            logger("arr_solver::tron done.");
        }
        else if (solver_obj)
        {
            logger("arr_solver::solver_obj calling solve()");

            solver_obj->solve(w);
            logger("arr_solver::solver_obj solved.");
        }
    }

    double solver_fun(arma::mat &w)
    {

        if (!done_init)
        {
            logger("arr_solver::solver_fun() calling init_prob()");
            init_prob();
        }
        if (fun_obj)
        {
            logger("arr_solver::solver_fun() calling fun_obj->fun()");
            return fun_obj->fun(w);
        }
        else if (solver_obj)
        {
            logger("arr_solver::solver_fun() calling solver_obj->solver_fun()");
            return solver_obj->solver_fun(w);
        }
        else
        {
            return 0;
        }
    }
};

class arr_base_IX : public function
{
public:
    typedef arr_prob_t prob_t;
    typedef arr_param_t param_t;

protected:
    const prob_t *prob;
    const param_t *param;

    // const reference to prob
    const arma::mat &Y;
    const arma::mat &H;
    const size_t &m;
    const size_t &n;
    const size_t &k;
    const double &lambdaI;
    const double &lambdaAR;

public:
    arr_base_IX(const prob_t *prob, const param_t *param) : prob(prob), param(param), Y(prob->Y), H(prob->H),
                                                            m(prob->m), n(prob->n), k(prob->k),
                                                            lambdaI(param->lambdaI), lambdaAR(param->lambdaAR) {}

    virtual void init() {}
    double fun(arma::mat &w);
    void grad(arma::mat &w, arma::mat &g);
    void Hv(arma::mat &s, arma::mat &Hs);
};

// AutoRegressive regularization + squared-L2 loss + full observation
// See arr_prob_t for the mathmatical definitions

class arr_ls_fY_IX : public arr_base_IX
{
public:
    typedef arr_base_IX base_t;
    typedef typename base_t::prob_t prob_t;
    typedef typename base_t::param_t param_t;

protected:
    double trYTY;
    arma::mat YH;  // m * k => Y * H
    arma::mat HTH; // k * k => H^T * H
    arma::mat WTW; // k * k => W^T * W

public:
    arr_ls_fY_IX(const prob_t *prob, const param_t *param) : base_t(prob, param)
    {
        trYTY = arma::dot(Y, Y);
        YH = arma::mat(m, k);
        HTH = arma::mat(k, k);
        WTW = arma::mat(k, k);
    }

    void init()
    {
        logger("arr_ls_fY_IX::init() ");

        trYTY = arma::dot(Y, Y);
        YH = Y * H;
        HTH = H.t() * H;
    }

    double fun(arma::mat &w);

    void grad(arma::mat &w, arma::mat &g);

    void Hv(arma::mat &s, arma::mat &Hs);
};
class l2r_autoregressive_solver
{
    const arma::mat &T; // m \times k
    const arma::uvec &lag_set;
    const size_t m;
    const size_t k;
    double lambda;
    size_t nr_threads;
    std::vector<arma::vec> univate_series_set;
    std::vector<arma::mat> Hessian_set;
    bool done_init;

public:
    l2r_autoregressive_solver(const arma::mat &T, const arma::uvec &lag_set, double lambda) : T(T),
                                                                                              lag_set(lag_set),
                                                                                              m(T.n_rows),
                                                                                              k(T.n_cols),
                                                                                              lambda(lambda)
    {
        nr_threads = 12;
        univate_series_set.resize(nr_threads, arma::vec(m));
        Hessian_set.resize(nr_threads, arma::mat(lag_set.size(), lag_set.size()));
    }

    void init_prob()
    {
    }

    double lagged_inner_product(const arma::vec &univate_series, size_t start, size_t end, size_t lag_1, size_t lag_2)
    {
        double ret = 0;
        for (size_t i = start; i < end; i++)
        {
            ret += univate_series.at(i - lag_1) * univate_series.at(i - lag_2);
        }
        return ret;
    }

    void solve(arma::mat &lag_val)
    {
        size_t start = lag_set.back();
        size_t end = m;
        logger("l2_autoregressive_solver::solve()");

#pragma omp parallel for schedule(static)
        for (size_t t = 0; t < k; t++)
        {
            int tid = omp_get_thread_num(); // thread ID
            arma::vec &univate_series = univate_series_set[tid];
            for (size_t i = 0; i < m; i++)
            {
                univate_series.at(i) = T.at(i, t);
            }
            arma::vec y = lag_val.col(t);
            arma::mat &Hessian = Hessian_set[tid];
            for (size_t i = 0; i < lag_set.size(); i++)
            {
                size_t lag_i = lag_set[i];
                y[i] = lagged_inner_product(univate_series, start, end, static_cast<size_t>(0), lag_i);
                for (size_t j = i; j < lag_set.size(); j++)
                {
                    size_t lag_j = lag_set[j];
                    double Hij = lagged_inner_product(univate_series, start, end, lag_i, lag_j);
                    Hessian.at(i, j) = Hij;
                }
            }
            for (size_t i = 0; i < lag_set.size(); i++)
            {
                for (size_t j = 0; j < i; j++)
                {
                    Hessian.at(i, j) = Hessian.at(j, i);
                }
                Hessian.at(i, i) += lambda;
            }


            lag_val.col(t) = arma::solve(Hessian, y);


        }
    }

    double fun(arma::mat &lag_val)
    {
        logger("l2_autoregressive_solver::fun()");

        size_t start = lag_set.back();
        size_t end = m;
        double loss = 0.0;
        for (size_t t = 0; t < k; t++)
        {
            double local_loss = 0.0;
            for (size_t i = start; i < end; i++)
            {
                double err = -T.at(i, t);
                for (size_t l = 0; l < lag_set.size(); l++)
                {
                    err += lag_val.at(l, t) * T.at(i - lag_set[l], t);
                }
                local_loss += err * err;
            }
            loss += local_loss;
        }
        loss *= 0.5;
        double reg = 0.5 * arma::dot(lag_val, lag_val);
        return reg * lambda + loss;
    }
};
void trmf_initialization(const trmf_prob_t &prob, const trmf_param_t &param, arma::mat &W, arma::mat &H, arma::mat &lag_val);
bool check_dimension(const trmf_prob_t &prob, const trmf_param_t &param, const arma::mat &W, const arma::mat &H, const arma::mat &lag_val);
void trmf_train(trmf_prob_t &prob, trmf_param_t &param, arma::mat &W, arma::mat &H, arma::mat &lag_val);
// arma::mat multi_pred(trmf_prob_t &prob, trmf_param_t &param, size_t window, size_t nb_windows);

arma::mat multi_pred(arma::mat& data, trmf_param_t &param, size_t window, size_t nb_windows, arma::uvec lagset, size_t rank);


