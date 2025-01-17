///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, INRIA,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
template <typename Scalar>
CostModelResidualTpl<Scalar>::CostModelResidualTpl(
    boost::shared_ptr<typename Base::StateAbstract> state,
    boost::shared_ptr<ActivationModelAbstract> activation,
    boost::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, activation, residual) {}

template <typename Scalar>
CostModelResidualTpl<Scalar>::CostModelResidualTpl(
    boost::shared_ptr<typename Base::StateAbstract> state,
    boost::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, residual) {}

template <typename Scalar>
CostModelResidualTpl<Scalar>::~CostModelResidualTpl() {}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calc(
    const boost::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
      std::cout << "residual cost" << std::endl;
  // Compute the cost residual
  std::cout << "residual_->calc" << std::endl;
  residual_->calc(data->residual, x, u);
  std::cout << "x: " << x << std::endl;
  std::cout << "u: " << u << std::endl;
  std::cout << "residual_->calc done" << std::endl;

  // Compute the cost
  std::cout << "activation_->calc" << std::endl;
  activation_->calc(data->activation, data->residual->r);
  std::cout << "activation_->calc done" << std::endl;
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calc(
    const boost::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  if (!is_rq && !is_rv) {
    data->activation->a_value = 0.;
    data->cost = 0.;
    return;  // do nothing
  }

  // Compute the cost residual
  residual_->calc(data->residual, x);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calcDiff(
    const boost::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and contact wrench cone residual
  // models
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton
  // approximation
  residual_->calcCostDiff(data, data->residual, data->activation);
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calcDiff(
    const boost::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  // Compute the derivatives of the activation and contact wrench cone residual
  // models
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  if (!is_rq && !is_rv) {
    data->Lx.setZero();
    data->Lxx.setZero();
    return;  // do nothing
  }
  residual_->calcDiff(data->residual, x);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton
  // approximation
  residual_->calcCostDiff(data, data->residual, data->activation, false);
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> >
CostModelResidualTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::print(std::ostream& os) const {
  os << "CostModelResidual {" << *residual_ << ", " << *activation_ << "}";
}

}  // namespace crocoddyl
