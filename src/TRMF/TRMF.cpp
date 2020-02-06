#include <../include/TRMF/TRMF.h>

using arma::mat;
using arma::uvec;
using arma::vec;

void logger(std::string text)
{
    if (VERBOSE)
    {
        std::cout << text << std::endl;
    }
}

struct l2r_ls_fY_IX_chol : public solver_t
{ // {{{
    const mat &Y;
    const mat &H;
    mat YH;
    mat HTH;
    mat kk_buf;
    double trYTY;
    double lambda;
    const size_t m, k;
    bool done_init;

    l2r_ls_fY_IX_chol(const mat &Y, const mat &H, double lambda) : Y(Y), H(H), trYTY(0), lambda(lambda), m(Y.n_rows), k(H.n_cols), done_init(false)
    {

        YH = mat(m, k);
        HTH = mat(k, k);
      
        logger("l2r_ls_fY_IX_chol::constructor()");

        trYTY = arma::dot(Y, Y);
    }

    void init_prob()
    {
        logger("l2r_ls_fY_IX_chol::init_prob()");

        YH = Y * H;
        HTH = H.t() * H;

        for (size_t t = 0; t < k; t++)
        {

            HTH.at(t, t) += lambda;
        }

        done_init = true;
    }

    void solve(arma::mat &W)
    {
        logger("l2r_ls_fY_IX_chol::solve()");

        if (!done_init)
        {
            init_prob();
        }
        W = YH;
        arma::mat Res = arma::solve(HTH, W.t());
        W = Res.t();        
        // HTH will be changed to a cholesky factorization after the above call.
        // Thus we need to flip done_init to false again
        done_init = false;
    }

    double solver_fun(arma::mat W)
    {
        logger("l2r_ls_fY_IX_chol::fun()");
        if (!done_init)
        {
            init_prob();
        }
        //TODO::reshape needed ?

        W.reshape(m, k);
        double obj = trYTY;
        kk_buf = W.t() * W;

        obj += arma::dot(kk_buf, HTH);
        obj -= 2.0 * arma::dot(W, YH);
        obj *= 0.5;
        return obj;
    }
}; // }}}

arr_solver::arr_solver(arr_prob_t *prob, arr_param_t *param) : prob(prob),
                                                               param(param),
                                                               tron_obj(NULL),
                                                               solver_obj(NULL),
                                                               done_init(false)
{
    if (prob->lag_set != NULL)
    {
        logger("setting tron solver tron_obj");
        logger("setting arr_ls_fY_IX fun_obj");

        fun_obj = new arr_ls_fY_IX(prob, param);
        tron_obj = new tron(*fun_obj, param->eps, param->eps_cg, param->max_tron_iter, param->max_cg_iter);
        solver_obj = NULL;
    }
    else
    {
        logger("setting chol solver_obj");

        fun_obj = NULL;
        tron_obj = NULL;
        solver_obj = new l2r_ls_fY_IX_chol(prob->Y, prob->H, param->lambdaI);
    }
}

double arr_base_IX::fun(mat &W)
{
    logger("arr_base_IX:: fun()");

    // W.reshape(m, k);
    double f = 0;
    logger("arr_base_IX:reshape() done.");

    if (lambdaI > 0)
    {
        f += 0.5 * lambdaI * arma::dot(W, W);
    }
    if (prob->lag_set != NULL and lambdaAR > 0)
    {
        const uvec &lag_set = (*(prob->lag_set));
        const mat &lag_val = *(prob->lag_val);
        size_t midx = lag_set.back(); // supposed to be the max index in lag_set
        double AR_val = 0;

#pragma omp parallel for reduction(+ \
                                   : AR_val)
        for (size_t i = midx; i < m; i++)
        {
            double tmp_val = 0;
            for (size_t t = 0; t < k; t++)
            {
                double residual = W.at(i, t);
                for (size_t l = 0; l < lag_set.n_rows; l++)
                {
                    size_t lag = lag_set[l];
                    residual -= lag_val.at(l, t) * W.at(i - lag, t);
                }
                tmp_val += residual * residual;
            }
            AR_val += tmp_val;
        }
        f += 0.5 * lambdaAR * AR_val;
    }
    logger("arr_base_IX:: fun() done.");
    // W = W.as_col();
    return f;
}

void arr_base_IX::grad(mat &W, mat &G)
{
    logger("arr_base_IX:: grad()");

    W.reshape(m, k);
    G = lambdaI * W;

    if (prob->lag_set != NULL and lambdaAR > 0)
    {
        const uvec &lag_set = *(prob->lag_set);
        const mat &lag_val = *(prob->lag_val);

        size_t midx = lag_set.back(); // supposed to be the max index in lag_set

#pragma omp parallel for
        for (size_t t = 0; t < k; t++)
        {
            for (size_t i = midx; i < m; i++)
            {
                double residual = W.at(i, t);
                for (size_t l = 0; l < lag_set.n_rows; l++)
                {
                    size_t lag = lag_set[l];
                    residual -= lag_val.at(l, t) * W.at(i - lag, t);
                }
                G.at(i, t) += lambdaAR * residual;
                for (size_t l = 0; l < lag_set.size(); l++)
                {
                    size_t lag = lag_set[l];
                    G.at(i - lag, t) -= lambdaAR * residual * lag_val.at(l, t);
                }
            }
        }
    }

    W = W.as_col();
    G = G.as_col();

    logger("arr_base_IX:: grad() done.");
}
void arr_base_IX::Hv(mat &S, mat &Hs)
{
    logger("arr_base_IX:: Hv()");

    S.reshape(m, k);

    Hs = lambdaI * S;

    if (lambdaAR > 0)
    {
        const uvec &lag_set = *(prob->lag_set);
        const mat &lag_val = *(prob->lag_val);
        size_t midx = lag_set.back();
#pragma omp parallel for
        for (size_t t = 0; t < k; t++)
        {
            for (size_t i = midx; i < m; i++)
            {
                double residual = S.at(i, t);
                for (size_t l = 0; l < lag_set.size(); l++)
                {
                    size_t lag = lag_set[l];
                    residual -= lag_val.at(l, t) * S.at(i - lag, t);
                }
                Hs.at(i, t) += lambdaAR * residual;
                for (size_t l = 0; l < lag_set.size(); l++)
                {
                    size_t lag = lag_set[l];
                    Hs.at(i - lag, t) -= lambdaAR * residual * lag_val.at(l, t);
                }
            }
        }
    }
    S = S.as_col();
    Hs = Hs.as_col();
    logger("arr_base_IX:: Hv() done.");
}
double arr_ls_fY_IX::fun(mat &W)
{
    logger("arr_ls_fY_IX:: fun()");
    W.reshape(m, k);
    double f = base_t::fun(W);

    f += 0.5 * trYTY;

    WTW = W.t() * W;

    f += 0.5 * arma::dot(WTW, HTH);
    f -= arma::dot(YH, W);
    W = W.as_col();
    logger("arr_ls_fY_IX:: fun() done.");

    return f;
}
void arr_ls_fY_IX::grad(mat &W, mat &G)
{
    logger("arr_ls_fY_IX:: grad()");

    base_t::grad(W, G);

    W.reshape(m, k);
    G.reshape(m, k); // view constructor

    G += -YH + W * HTH;

    G = G.as_col();

    W = W.as_col();

    logger("arr_ls_fY_IX:: grad() done");
}
void arr_ls_fY_IX::Hv(mat &S, mat &Hs)
{
    logger("arr_ls_fY_IX:: Hv()");

    base_t::Hv(S, Hs);

    S.reshape(m, k); // view constructor
    Hs.reshape(m, k);
    Hs += S * HTH;

    S = S.as_col();
    Hs = Hs.as_col();
    logger("arr_ls_fY_IX:: Hv() done.");
}
arma::mat multi_pred(arma::mat& data, trmf_param_t &param, size_t window, size_t nb_windows, arma::uvec lagset, size_t rank)
{


   
    size_t cols = data.n_cols;
    

    arma::mat full = data;
    arma::mat pred((window * nb_windows), cols, arma::fill::zeros);

    arma::mat W;
    arma::mat H;
    arma::mat lag_val;
    double total;
   

    size_t trn_start, trn_end;
    for (size_t i = 0; i < nb_windows; i++)
    {
        trn_start = 0;
        trn_end = data.n_rows - (nb_windows - i) * window;
        trn_end--;

        
        arma::mat Y = full.rows(arma::span(trn_start, trn_end));
        arma::mat Yt= Y.t();
        trmf_prob_t prob(Y,Yt,lagset,rank);
        size_t rows = Y.n_rows;
        arma::mat newW(rows + window, rank, arma::fill::zeros);

        if (i == 0)
        {
            trmf_initialization(prob, param, W, H, lag_val);
            if (not check_dimension(prob, param, W, H, lag_val))
                break;
        }
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        trmf_train(prob, param, W, H, lag_val);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference = " 
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
        << "[ms]" << std::endl;
        total += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        
        //latent forecast
        newW.rows(0, rows  - 1) = W.rows(0, rows - 1);
        
      
        
        for (size_t j = rows; j < rows + window; j++)
        {
            arma::mat temp = newW.rows(j - prob.lag_set) % lag_val;
           
            
            newW.row(j) = arma::sum(temp);
        }
     
        
        
        arma::mat doty = (newW* H.t());
        pred.rows(arma::span(i * window, (i + 1) * window -1)) =doty.rows(doty.n_rows - window, doty.n_rows - 1) ;
        W = newW;
    }
    std::cout<< "total " << total<<std::endl;
    
    return pred;
}

void trmf_train(trmf_prob_t &prob, trmf_param_t &param, mat &W, mat &H, mat &lag_val)
{
    uvec &lag_set = (prob.lag_set);
    mat &Y = (prob.Y);
    mat &Yt = (prob.Yt); // transpose of Y

    logger("Training trmf");

    if (param.max_tron_iter > 1)
    {
        param.max_cg_iter *= param.max_tron_iter;
        param.max_tron_iter = 1;
    }

    arr_prob_t subprob_W(Y, H, &lag_set, &lag_val);
    arr_prob_t subprob_H(Yt, W, NULL, NULL);

    arr_solver W_solver(&subprob_W, &param);
    arr_solver H_solver(&subprob_H, &param);

    l2r_autoregressive_solver LV_solver(W, lag_set, param.lambdaLag);

    for (int iter = 1; iter <= param.max_iter; iter++)
    {

        if ((iter % param.period_H) == 0)
        {
            logger("Init H");
            H_solver.init_prob();
            logger("Solving H");
            H_solver.solve(H);
            logger("H solved");
        }

        if ((iter % param.period_W) == 0)
        {
            logger("Init W");
            W_solver.init_prob();
            logger("Solving W");
            W_solver.solve(W);


            logger("W solved");
        }

        if ((iter % param.period_Lag) == 0)
        {

            logger("Init theta");
            LV_solver.init_prob();
            logger("Solving theta");
            LV_solver.solve(lag_val);
            logger("Theta solved");
        }
    }

}
void trmf_initialization(const trmf_prob_t &prob, const trmf_param_t &param, mat &W, mat &H, mat &lag_val)
{
    logger("Trmf initialisation");

    size_t m = prob.m;
    size_t n = prob.n;
    size_t k = prob.k;

    W = mat(m, k, arma::fill::randn);
    H = mat(n, k, arma::fill::randn);

    lag_val = mat(prob.lag_set.n_rows, k, arma::fill::randn);


}
bool check_dimension(const trmf_prob_t &prob, const trmf_param_t &param, const mat &W, const mat &H, const mat &lag_val)
{
    bool pass = true;
    if (prob.m != W.n_rows)
    {
        std::cerr << "[ERR MSG]: Y.rows (" << prob.m << ") != W.rows (" << W.n_rows << ")\n"
                  << std::endl;
        pass = false;
    }
    if (prob.n != H.n_rows)
    {
        std::cerr << "[ERR MSG]: Y.cols (" << prob.n << ") != H.rows (" << H.n_rows << ")\n"
                  << std::endl;
        pass = false;
    }
    if (W.n_cols != H.n_cols)
    {
        std::cerr << "[ERR MSG]: W.cols (" << W.n_cols << ") != H.cols (" << H.n_cols << ")\n"
                  << std::endl;
        pass = false;
    }
    if (prob.lag_set.size() != lag_val.n_rows)
    {
        std::cerr << "[ERR MSG]: lag_set.size(" << prob.lag_set.size() << ") != lag_val.rows(" << lag_val.n_rows << ")\n"
                  << std::endl;
        pass = false;
    }
    if (W.n_cols != lag_val.n_cols)
    {
        std::cerr << "[ERR MSG]: W.cols(" << W.n_cols << ") != lag_val.cols( " << lag_val.n_cols << ")\n"
                  << std::endl;
        pass = false;
    }
    if (pass)
    {
        logger("Check dimension ok.");
    }

    return pass;
}
