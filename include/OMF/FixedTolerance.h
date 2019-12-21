#pragma once
#include "TemplateOMF.h"

class FixedTolerance : public TemplateOMF {
public:
	explicit FixedTolerance(arma::mat &data, double epsilon = 1e-2, double U_penalty = 1.);

protected:
	double m_epsilon;
	double m_penU;

	void expectation(size_t t) override;
	void prediction2(size_t t) override;
};

