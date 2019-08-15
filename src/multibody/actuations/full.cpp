///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actuations/full.hpp"

namespace crocoddyl {

ActuationModelFull::ActuationModelFull(StateMultibody& state) : ActuationModelAbstract(state, state.get_nv()) {
  pinocchio::JointModelFreeFlyer ff_joint;
  assert(state.get_pinocchio().joints[1].shortname() == ff_joint.shortname() &&
         "ActuationModelFull: the first joint cannot be free-flyer");
  if (state.get_pinocchio().joints[1].shortname() == ff_joint.shortname()) {
    std::cout << "Warning: the first joint cannot be a free-flyer" << std::endl;
  }
}

ActuationModelFull::~ActuationModelFull() {}

void ActuationModelFull::calc(const boost::shared_ptr<ActuationDataAbstract>& data,
                              const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(u.size() == nu_ && "ActuationModelFull::calc: u has wrong dimension");
  data->a.diagonal() = u;
}

void ActuationModelFull::calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data,
                                  const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, u);
  }
  // The derivatives has constant values which were set in createData.
}

boost::shared_ptr<ActuationDataAbstract> ActuationModelFull::createData() {
  boost::shared_ptr<ActuationDataAbstract> data = boost::make_shared<ActuationDataAbstract>(this);
  data->Au.diagonal().fill(1);
  return data;
}

}  // namespace crocoddyl
