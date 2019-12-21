#include "TemplateOTS.h"

class OTS_gsr : public TemplateOTS
{
public:
    explicit OTS_gsr(arma::mat &data, size_t d = 15, double eta = 1);

protected:
    arma::vec vec_forecast(arma::vec const &d) override;

private:
    arma::mat inner_product(arma::vec const &y, size_t d);
	size_t m_lag;
    
};
