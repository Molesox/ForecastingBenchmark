#pragma once
#include <armadillo>
class TemplateOMF
{
public:
	explicit TemplateOMF(arma::mat &data, size_t order = 24, size_t dim = 5,
		double V_penalty = 1e-4, double reg = 1., size_t max_iter = 15);

	virtual ~TemplateOMF() = default;

	arma::mat forecast();//cannot be overrided

	std::string m_name;

protected:

	arma::mat &m_data;

	//Scalar attributs
	size_t m_order;
	size_t m_dim;
	double m_penV;
	double m_reg;
	size_t m_max_iter;

	//Matricial attributs
	arma::mat m_pred;
	arma::mat m_U;
	arma::mat m_hU;
	arma::mat m_V;
	arma::mat m_BLK;
	arma::mat m_lhs;
	arma::mat m_W;
	arma::mat m_eye;

	//Vector attributs
	arma::vec m_hV;
	arma::vec m_nV;
	arma::vec m_obs;
	arma::vec m_rhs;


	virtual void prediction1(size_t t);//may be overrided
	virtual void prediction2(size_t t);
	virtual void prediction3(size_t t);

	virtual void expectation(size_t t) = 0;//must be overrided

	virtual void maximisation();//may be overrided


private:

};

