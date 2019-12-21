#pragma once
#include "TemplateOATS.h"

class OATS_ogd : public TemplateOATS
{
public:
   explicit OATS_ogd(arma::mat &data, int m = 5, int k = 5, double eta = 1);
   
protected:
   arma::vec vec_forecast(arma::vec const &x) override;
    arma::rowvec m_lambda;
};
