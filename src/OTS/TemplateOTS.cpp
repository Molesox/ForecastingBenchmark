#include "OTS/TemplateOTS.h"



using arma::mat;

TemplateOTS::TemplateOTS(arma::mat &data, double eta0) :
	m_data(data), m_eta(eta0) {

	m_N = data.n_rows;
	m_pred = mat(m_data.n_rows, m_data.n_cols, arma::fill::zeros);

	m_err = arma::colvec(m_N, arma::fill::zeros);
}

arma::mat TemplateOTS::forecast() {
	for (size_t i = 0; i < m_data.n_cols; ++i) {
		m_pred.col(i) = vec_forecast(m_data.col(i));
		std::cout << "." << std::flush;
	}
	std::cout << std::endl;
	return m_pred;
}