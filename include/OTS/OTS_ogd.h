#pragma once
#include "TemplateOTS.h"

class OTS_ogd : public TemplateOTS {
public:
	explicit OTS_ogd(arma::mat &data, size_t ar_order = 15, double eta = 1);

protected:
	arma::vec vec_forecast(arma::vec const &d)override;

	//Attributs
	size_t m_order;
	arma::mat m_alpha;


};
