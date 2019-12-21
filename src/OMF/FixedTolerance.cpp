#include "OMF/FixedTolerance.h"

FixedTolerance::FixedTolerance(arma::mat &data, double epsilon, double U_penalty) :
	TemplateOMF(data),
	m_epsilon(epsilon),
	m_penU(U_penalty) {
	m_name = "FT";

}

void FixedTolerance::expectation(size_t t) {

	static arma::vec polynome = arma::vec(3);
	static double a1, a2, a3, a4, a5;

	if (t == 1)m_hU.zeros();

	for (size_t i = 0; i < m_max_iter; ++i) {

		m_V = arma::solve((m_penV * m_eye + m_U * m_U.t()), (m_penV * m_hV + m_U * m_obs));

		a1 = arma::sum(m_obs % (m_hU.t() * m_V));
		a2 = pow((arma::norm(m_V)), 2);
		a3 = pow((arma::norm(m_obs)), 2);
		a4 = pow(arma::norm(m_hU.t() * m_V), 2);
		a5 = m_epsilon * m_epsilon - a3;

		polynome(0) = -(a3 + a5) * a2 * a2;
		polynome(1) = -2 * (a3 + a5) * a2;
		polynome(2) = a4 - 2 * a1 - a5;

		m_penU = 1. / arma::max(arma::real(arma::roots(polynome)));

		m_U = arma::solve(m_eye * m_penU + m_V * m_V.t(), m_penU * m_hU + m_V * m_obs.t());
	}

}

void FixedTolerance::prediction2(size_t t) {

	m_pred.col((t - 1)) = m_U.t() * m_V;
	m_hV.zeros();

}