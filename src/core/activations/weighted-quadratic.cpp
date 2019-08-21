///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/weighted-quadratic.hpp"

namespace crocoddyl {

ActivationModelWeightedQuad::ActivationModelWeightedQuad(const Eigen::VectorXd& weights)
    : ActivationModelAbstract((unsigned int)weights.size()), weights_(weights) {}

ActivationModelWeightedQuad::~ActivationModelWeightedQuad() {}

void ActivationModelWeightedQuad::calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                                       const Eigen::Ref<const Eigen::VectorXd>& r) {
  assert(r.size() == nr_ && "ActivationModelWeightedQuad::calc: r has wrong dimension");
  boost::shared_ptr<ActivationDataWeightedQuad> d = boost::static_pointer_cast<ActivationDataWeightedQuad>(data);

  d->Wr = weights_.cwiseProduct(r);
  data->a_value = 0.5 * r.transpose() * d->Wr;
}

void ActivationModelWeightedQuad::calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                                           const Eigen::Ref<const Eigen::VectorXd>& r, const bool& recalc) {
  assert(r.size() == nr_ && "ActivationModelWeightedQuad::calcDiff: r has wrong dimension");
  if (recalc) {
    calc(data, r);
  }

  boost::shared_ptr<ActivationDataWeightedQuad> d = boost::static_pointer_cast<ActivationDataWeightedQuad>(data);
  data->Ar = d->Wr;
  data->Arr.diagonal() = weights_;
}

boost::shared_ptr<ActivationDataAbstract> ActivationModelWeightedQuad::createData() {
  return boost::make_shared<ActivationDataWeightedQuad>(this);
}

const Eigen::VectorXd& ActivationModelWeightedQuad::get_weights() const { return weights_; }

}  // namespace crocoddyl
