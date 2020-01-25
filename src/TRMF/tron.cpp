#include <../include/TRMF/tron.h>
using arma::mat;
using arma::vec;

void log1(std::string text)
{
    if (VERBOSE)
    {
        std::cout << text << std::endl;
    }

}

tron::tron(function &fun, double eps, double eps_cg, size_t maxiter, size_t maxiter_cg) : m_fun(fun),
                                                                                          m_eps(eps),
                                                                                          m_eps_cg(eps_cg),
                                                                                          m_maxiter(maxiter),
                                                                                          m_maxiter_cg(maxiter_cg)

{
    //    g = mat(2,2);
}
tron::~tron()
{
}

void tron::trust_region(mat &w, bool set2zero)
{
    log1("TRON:: Trust region on ...");
    // mat temp = w;
    // temp.reshape(100,5);
    // std::cout<< "before fun() = " <<std::endl;
    // std::cout << "w = (" << w.n_rows << "," << w.n_cols << ")" << std::endl;
    // std::cout << temp <<std::endl;
    
    
    // Parameters for updating the iterates.
    double eta0 = 1e-4;
    double eta1 = 0.25;
    double eta2 = 0.75;

    // Parameters for updating the trust region size delta.
    double sigma1 = 0.25;
    double sigma2 = 0.5;
    double sigma3 = 4;

    int i;
    size_t cg_iter;
    double delta;
    double snorm;

    double one = 1.0;

    double alpha;
    double f;
    double fnew;
    double prered;
    double actred;
    double gs;
    size_t search = 1;
    size_t iter = 1;

    if (set2zero)
    {
        w.zeros();
        log1("TRON:: ... setting to 0 ...");
    }

    f = m_fun.fun(w);
    log1("TRON::fun() done.");

    // temp = w;
    // temp.reshape(100,5);
    // std::cout<< "after fun()" <<std::endl;
    // std::cout << "temp = (" << temp.n_rows << "," << temp.n_cols << ")" << std::endl;
    // std::cout << temp <<std::endl;
    
    

    m_fun.grad(w, g);
    log1("TRON::grad() done.");

    
    // temp = w;
    // temp.reshape(100,5);
    // std::cout<< "after grad()" <<std::endl;
    // std::cout << "temp = (" << temp.n_rows << "," << temp.n_cols << ")" << std::endl;
    // std::cout << temp <<std::endl;
    

    delta = sqrt(arma::dot(g, g));
    double gnorm1 = delta;
    double gnorm = gnorm1;

    if (gnorm <= m_eps * gnorm1)
        search = 0;

    iter = 1;
    bool printed = false;
    while (iter <= m_maxiter and search)
    {
        double cg_rnorm = 0;

        if(not printed){
            std::cout<< "BEFORE TRCG" <<std::endl;
            
            mat temp = g;
            temp.reshape(100,5);
            std::cout << "g = (" << g.n_rows << "," << g.n_cols << ")" << std::endl;
            std::cout << temp <<std::endl;
            temp = s;
            temp.reshape(100,5);
            std::cout << "s = (" << s.n_rows << "," << s.n_cols << ")" << std::endl;
            std::cout << temp <<std::endl;
            temp = r;
            temp.reshape(100,5);
            std::cout << "r = (" << r.n_rows << "," << r.n_cols << ")" << std::endl;
            std::cout << temp <<std::endl;
            
        }

        cg_iter = trcg(delta, g, s, r, cg_rnorm);


        if(not printed){
            std::cout<< "after TRCG" <<std::endl;
            
            mat temp = g;
            temp.reshape(100,5);
            std::cout << "g = (" << g.n_rows << "," << g.n_cols << ")" << std::endl;
            std::cout <<temp <<std::endl;
            temp = s;
            temp.reshape(100,5);
            std::cout << "s = (" << s.n_rows << "," << s.n_cols << ")" << std::endl;
            std::cout << temp <<std::endl;
            temp = r;
            temp.reshape(100,5);
            std::cout << "r = (" << r.n_rows << "," << r.n_cols << ")" << std::endl;
            std::cout <<temp <<std::endl;
            printed = true;
        }

        log1("TRON:: ... conjugate grad ...");

        w_new = w;
        w_new += s;

        gs = arma::dot(g, s);

        prered = -0.5 * (gs - arma::dot(s, r));
        fnew = m_fun.fun(w_new);

        // Compute the actual reduction.
        actred = f - fnew;

        // On the first iteration, adjust the initial step bound.
        snorm = sqrt(arma::dot(s, s));
        if (iter == 1)
            delta = std::min(delta, snorm);

        // Compute prediction alpha*snorm of the step.
        if (fnew - f - gs <= 0)
        {
            alpha = sigma3;
        }
        else
        {
            alpha = std::max(sigma1, -0.5 * (gs / (fnew - f - gs)));
        }

        // Update the trust region bound according to the ratio of actual to predicted reduction.
        if (actred < eta0 * prered)
        {
            delta = std::min(std::max(alpha, sigma1) * snorm, sigma2 * delta);
        }
        else if (actred < eta1 * prered)
        {
            delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma2 * delta));
        }
        else if (actred < eta2 * prered)
        {
            delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma3 * delta));
        }
        else
        {
            delta = std::max(delta, std::min(alpha * snorm, sigma3 * delta));
        }

        if (actred > eta0 * prered)
        {
            iter++;
            w = w_new;
            f = fnew;
            m_fun.grad(w, g);
            gnorm = sqrt(arma::dot(g, g));

            if (gnorm <= m_eps * gnorm1)
                break;
        }
        if (f < -1.0e+32)
        {
            std::cerr << "turst_region, WARNING: f < -1.0e+32\n"
                      << std::endl;
            break;
        }
        if (fabs(actred) <= 0 and prered <= 0)
        {
            std::cerr << "trust_region, WARNING: actred and prered <= 0\n"
                      << std::endl;
            break;
        }
        if (fabs(actred) <= 1.0e-12 * fabs(f) and fabs(prered) <= 1.0e-12 * fabs(f))
        {
            std::cerr << "trust_region, WARNING: actred and prered too small\n";
            break;
        }
    }
    log1("TRON:: ... trust region done.");
}

size_t tron::trcg(double delta, mat &g, mat &s, mat &r, double cg_rnorm)
{
    int i;
    double one = 1;
    double rTr;
    double rnewTrnew;
    double alpha;
    double beta;
    double cgtol;

    s.zeros(g.size());
    r = -g;
    d = r;

    cgtol = m_eps_cg * sqrt(arma::dot(g, g));
    size_t cg_iter = 0;

    rTr = arma::dot(r, r);
    log1("TRON:: ... trcg starting iter ...");

    while (true)
    {
        cg_rnorm = sqrt(rTr);
        if (cg_rnorm <= cgtol)
            break;

        if (m_maxiter_cg > 0 and cg_iter >= m_maxiter_cg)
            break;

        cg_iter++;
        m_fun.Hv(d, Hd);

        alpha = rTr / arma::dot(d, Hd);
        s += alpha * d;

        alpha *= -1.;
        r += alpha * Hd;

        rnewTrnew = arma::dot(r, r);
        beta = rnewTrnew / rTr;

        double tmp = beta - 1.0;
        d += tmp * d;
        d += r;
        rTr = rnewTrnew;
    }
    log1("TRON:: ... trcg done. ");

    return cg_iter;
}