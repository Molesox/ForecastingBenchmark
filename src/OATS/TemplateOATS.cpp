#include "OATS/TemplateOATS.h"

TemplateOATS::TemplateOATS(arma::mat &data, int m, int k, double eta)
 : m_data(data), m_m(m), m_k(k), m_eta(eta)
{
    m_pred = arma::mat(m_data.n_rows, m_data.n_cols,arma::fill::zeros);

}

arma::mat TemplateOATS::forecast(){
	for (size_t i = 0; i < m_data.n_cols; ++i) {
		m_pred.col(i) = vec_forecast(m_data.col(i));
		std::cout << "." << std::flush;
	}
	std::cout << std::endl;
	return m_pred;
}