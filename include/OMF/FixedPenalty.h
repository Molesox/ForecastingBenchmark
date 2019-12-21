#pragma once
#include "TemplateOMF.h"

class FixedPenalty : public TemplateOMF {
public:
	explicit FixedPenalty(arma::mat &data, double U_penalty = 1.);

protected:

	double m_penU;

	void expectation(size_t t) override;

};
