#pragma once
#include <armadillo>

class TemplateOATS
{
public:
	explicit TemplateOATS(arma::mat &data, int m = 5, int k = 5, double eta = 1);
	virtual ~TemplateOATS() =default;
	
	arma::mat forecast();
	

	std::string m_name;

protected:
	virtual arma::vec  vec_forecast(arma::vec const &to_pred) = 0;
	arma::mat &m_data;
	int m_m;
	int m_k;
	double m_eta;

	arma::mat m_pred;
};
