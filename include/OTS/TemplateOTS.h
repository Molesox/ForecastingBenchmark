#pragma once
#include <armadillo>

class TemplateOTS {

public:

	explicit TemplateOTS(arma::mat &data, double eta0 = 1);
	virtual ~TemplateOTS() = default;

	arma::mat forecast();
	std::string m_name;
protected:

	virtual arma::vec vec_forecast(arma::vec const &to_pred) = 0;

	//Attributs
	arma::mat &m_data;


	size_t m_N;
	double m_eta;

	arma::vec m_err;
	arma::mat m_pred;

private:


};