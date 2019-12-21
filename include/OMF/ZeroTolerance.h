#pragma once
#include "TemplateOMF.h"

class ZeroTolerance : public TemplateOMF {
public:
	explicit ZeroTolerance(arma::mat &data);

protected:
	void expectation(size_t t) override;

};
