#include "OMF/FixedPenalty.h"

FixedPenalty::FixedPenalty(arma::mat &data, double U_penalty) :
	TemplateOMF(data), m_penU(U_penalty) {
	m_name = "FP";
}

void FixedPenalty::expectation(size_t t) {

	if (t == 1)m_hU.zeros();
	for (size_t i = 0; i < m_max_iter; ++i) {

		m_V = arma::solve(((m_penV * m_eye) + (m_U * m_U.t())), ((m_penV * m_hV) + (m_U * m_obs)));
		m_U = arma::solve((m_penU * m_eye + m_V * m_V.t()), (m_penU * m_hU + m_V * m_obs.t()));
	}
}
