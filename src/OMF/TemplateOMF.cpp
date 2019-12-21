#include "OMF/TemplateOMF.h"

using arma::mat;
using arma::vec;//column vector

TemplateOMF::TemplateOMF(mat &data, size_t order, size_t dim,
	double V_penalty, double reg, size_t max_iter) : m_data(data),
	m_order(order),
	m_dim(dim),
	m_penV(V_penalty),
	m_reg(reg),
	m_max_iter(max_iter) {

	if (m_data.n_cols < m_data.n_rows) {
		std::cerr << "TemplateOMF.cpp: time series matrix"
			" seems to be transposed." << std::endl;
	}

	//Initialisation:

	m_V = vec(m_dim, arma::fill::randu);//random init
	m_hV = vec(m_dim, arma::fill::zeros);
	m_rhs = vec(m_order, arma::fill::zeros);
	m_W = vec(m_order, arma::fill::zeros);

	m_pred = mat(m_data.n_rows, m_data.n_cols, arma::fill::zeros);
	m_U = mat(m_dim, m_data.n_rows, arma::fill::randu);//random init
	m_hU = mat(m_dim, m_data.n_rows, arma::fill::zeros);
	m_BLK = mat(m_dim, m_order, arma::fill::zeros);
	m_lhs = mat(m_order, m_order, arma::fill::eye);

	m_lhs.for_each([reg](arma::mat::elem_type &val) { val *= reg; });//another way to do it?

	m_V.for_each([dim](arma::mat::elem_type &val) { val *= 2. / (double)dim; });
	m_U.for_each([dim](arma::mat::elem_type &val) { val *= 2. / (double)dim; });

	m_eye = arma::mat(m_dim, m_dim, arma::fill::eye);

}

arma::mat TemplateOMF::forecast() {

	size_t T = m_data.n_cols;
	for (size_t t = 1; t <= T; ++t) {

		if (t == 1) prediction1(t);
		if (t > 1 and t <= m_order + 1) prediction2(t);
		if (t > m_order + 1) prediction3(t);

		m_obs = m_data.col((t - 1));
		m_hU = m_U;

		expectation(t);

		if (t >= m_order + 1) maximisation();

		m_BLK = arma::join_horiz(m_V, m_BLK.cols(0, m_order - 2));
	}

	return m_pred;
}

void TemplateOMF::prediction1(size_t t) {

	m_pred.col((t - 1)) = arma::colvec(m_pred.n_rows, arma::fill::zeros);
	m_hV.zeros();
}

void TemplateOMF::prediction2(size_t t) {

	m_pred.col((t - 1)) = m_U.t() * m_V;
	m_hV = m_V;
}

void TemplateOMF::prediction3(size_t t) {

	m_nV = m_BLK * m_W;
	m_pred.col((t - 1)) = m_U.t() * m_nV;
	m_hV = m_nV;
}

void TemplateOMF::maximisation() {

	m_lhs += m_BLK.t() * m_BLK;
	m_rhs += m_BLK.t() * m_V;
	m_W = arma::solve(m_lhs, m_rhs);
}