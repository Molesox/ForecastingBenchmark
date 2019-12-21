#include "OMF/ZeroTolerance.h"

ZeroTolerance::ZeroTolerance(arma::mat &data) : TemplateOMF(data) {
	m_name = "ZT";
}

void ZeroTolerance::expectation(size_t t) {

	arma::colvec lambda;
	static arma::mat repV;

	for (size_t i = 0; i < m_max_iter; ++i) {

		m_V = arma::solve((m_penV * m_eye + m_U * m_U.t()), (m_penV * m_hV + m_U * m_obs));
		lambda = (m_hU.t() * m_V - m_obs) / arma::dot(m_V, m_V);
		repV = arma::repmat(m_V, 1, m_data.n_rows);
		m_U = m_hU - repV.each_row() % lambda.t();

	}

}
