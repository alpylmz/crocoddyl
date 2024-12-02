///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/intro.hpp"

#include <iostream>

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

SolverIntro::SolverIntro(boost::shared_ptr<ShootingProblem> problem)
    : SolverFDDP(problem),
      eq_solver_(LuNull),
      th_feas_(1e-4),
      rho_(0.3),
      upsilon_(0.),
      zero_upsilon_(false) {
  const std::size_t T = problem_->get_T();
  Hu_rank_.resize(T);
  KQuu_tmp_.resize(T);
  YZ_.resize(T);
  Hy_.resize(T);
  Qz_.resize(T);
  Qzz_.resize(T);
  Qxz_.resize(T);
  Quz_.resize(T);
  kz_.resize(T);
  Kz_.resize(T);
  ks_.resize(T);
  Ks_.resize(T);
  QuuinvHuT_.resize(T);
  Qzz_llt_.resize(T);
  Hu_lu_.resize(T);
  Hu_qr_.resize(T);
  Hy_lu_.resize(T);

  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    Hu_rank_[t] = nh;
    KQuu_tmp_[t] = Eigen::MatrixXd::Zero(ndx, nu);
    YZ_[t] = Eigen::MatrixXd::Zero(nu, nu);
    Hy_[t] = Eigen::MatrixXd::Zero(nh, nh);
    Qz_[t] = Eigen::VectorXd::Zero(nh);
    Qzz_[t] = Eigen::MatrixXd::Zero(nh, nh);
    Qxz_[t] = Eigen::MatrixXd::Zero(ndx, nh);
    Quz_[t] = Eigen::MatrixXd::Zero(nu, nh);
    kz_[t] = Eigen::VectorXd::Zero(nu);
    Kz_[t] = Eigen::MatrixXd::Zero(nu, ndx);
    ks_[t] = Eigen::VectorXd::Zero(nh);
    Ks_[t] = Eigen::MatrixXd::Zero(nh, ndx);
    QuuinvHuT_[t] = Eigen::MatrixXd::Zero(nu, nh);
    Qzz_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nh);
    Hu_lu_[t] = Eigen::FullPivLU<Eigen::MatrixXd>(nh, nu);
    Hu_qr_[t] = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(nu, nh);
    Hy_lu_[t] = Eigen::PartialPivLU<Eigen::MatrixXd>(nh);
  }
}

SolverIntro::~SolverIntro() {}

bool SolverIntro::solve(const std::vector<Eigen::VectorXd>& init_xs,
                        const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t maxiter, const bool is_feasible,
                        const double init_reg) {
  START_PROFILER("SolverIntro::solve");
  if (problem_->is_updated()) {
    resizeData();
    // we shouldn't come here
    std::cout << "SolverIntro::solve: problem is updated" << std::endl;
    exit(1);
  }
  xs_try_[0] =
      problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible
  // std::cout << "xs_try_[0] = " << xs_try_[0].transpose() << std::endl;
  // 9.63268e-05 1 2.618 -1.5707 -1 9.26536e-05 2.618 1 0 0 0 0 0 0  0
  
  // for some reason the only thing setCandidate does is setting the trajectory's final state = 0
  //std::cout << "first init_xs: " << init_xs[0].transpose() << std::endl;
  //std::cout << "first init_xs size: " << init_xs.size() << std::endl; // always 0
  setCandidate(xs_try_, init_us, is_feasible);
  std::cout << "xs_" << std::endl;
  for (std::size_t i = 0; i < xs_.size(); ++i) {
    std::cout << xs_[i].transpose() << std::endl;
  }
  //std::cout << "first init_xs size: " << init_xs.size() << std::endl; // always 0
  //std::cout << "second init_xs: " << init_xs[0].transpose() << std::endl; // seg fault because size is 0



  /*
  if (std::isnan(init_reg)) {
    std::cout << "we should be here" << std::endl;
    preg_ = reg_min_;
    dreg_ = reg_min_;
  } else {
    std::cout << "Shouldn't be here 1" << std::endl;
    exit(2);
    preg_ = init_reg;
    dreg_ = init_reg;
  }
  */
  preg_ = dreg_ = 1.0; // my magic number
  dreg_ = 2349823784932; // I guess this is not used? Need to check

  was_feasible_ = false;
  if (zero_upsilon_) {
    upsilon_ = 0.;
  }
  else{
    if(upsilon_ != 0){
      std::cout << "Something going on, upsilon should be zero" << std::endl;
      std::cout << "upsilon_ = " << upsilon_ << std::endl;
      exit(3);
    }
  }
  std::cout << "upsilon_ = " << upsilon_ << std::endl;

  bool recalcDiff = true;
  // std::cout << "maxiter: " << maxiter << std::endl;
  // maxiter is 100 by default, not related with T in python
  int new_maxiter = 1; // 10 is better, but this also gets close to the goal without crazy movements
  for (iter_ = 0; iter_ < new_maxiter; ++iter_) {
    std::cout << "first loop" << std::endl;
    std::cout << "preg_ " << preg_ << std::endl;
    std::cout << "dreg_ " << dreg_ << std::endl;
    while (true) {
      std::cout << "first inner loop" << std::endl;
      try {
        computeDirection(recalcDiff);
        // here, calcDiff works:
        /*
        integratedactionmodeleuler
        controlparametrizationmodelpolyzero
        fullactuation
        cost model sum
        residual cost
        residualmodelframeplacementtpl
        residual cost
        residual cost
        residualmodelstatetpl
        constraint model manager
        */
      } catch (std::exception& e) {
        std::cout << "Exception shouldn't happen in compute direction" << std::endl;
        exit(5);
        recalcDiff = false;
        increaseRegularization();
        if (preg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }
    updateExpectedImprovement(); // what is the point of this?
    expectedImprovement(); // what is the point of this?

    // Update the penalty parameter for computing the merit function and its
    // directional derivative For more details see Section 3 of "An Interior
    // Point Algorithm for Large Scale Nonlinear Programming"
    if (hfeas_ != 0 && iter_ != 0) {
      std::cout << "point of this? 1" << std::endl; // shouldn't happen, don't know why
      exit(6);
      upsilon_ =
          std::max(upsilon_, (d_[0] + .5 * d_[1]) / ((1 - rho_) * hfeas_));
    }

    // We need to recalculate the derivatives when the step length passes
    recalcDiff = false;
    for (std::vector<double>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
          std::cout << "second loop" << std::endl;
      steplength_ = *it;
      try {
        dV_ = tryStep(steplength_);
        dfeas_ = hfeas_ - hfeas_try_;
        dPhi_ = dV_ + upsilon_ * dfeas_;
      } catch (std::exception& e) {
        std::cout << "Exception shouldn't happen in tryStep" << std::endl;
        exit(7);
        continue;
      }
      expectedImprovement();
      dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);
      dPhiexp_ = dVexp_ + steplength_ * upsilon_ * dfeas_;
      if (dPhiexp_ >= 0) {  // descend direction
        if (std::abs(d_[0]) < th_grad_ || dPhi_ > th_acceptstep_ * dPhiexp_) { // for some reason, in my experiment, I only see this if
          std::cout << "if num 2" << std::endl;
          was_feasible_ = is_feasible_;
          setCandidate(xs_try_, us_try_, (was_feasible_) || (steplength_ == 1));
          cost_ = cost_try_;
          hfeas_ = hfeas_try_;
          merit_ = cost_ + upsilon_ * hfeas_;
          recalcDiff = true;
          break;
        }
      } else {  // reducing the gaps by allowing a small increment in the cost
                // value
        if (dV_ > th_acceptnegstep_ * dVexp_) {
          std::cout << "if num 3" << std::endl;
          exit(8);
          was_feasible_ = is_feasible_;
          setCandidate(xs_try_, us_try_, (was_feasible_) || (steplength_ == 1));
          cost_ = cost_try_;
          hfeas_ = hfeas_try_;
          merit_ = cost_ + upsilon_ * hfeas_;
          recalcDiff = true;
          break;
        }
        else{
          std::cout << "if num 4" << std::endl;
          exit(9);
        }
      }
    }

    stoppingCriteria(); // this updates stop_ variable in solver-base.hpp
    
    // No callbacks are used for now!
    const std::size_t n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      std::cout << "Do we have callbacks?" << std::endl;
      exit(8);
      CallbackAbstract& callback = *callbacks_[c];
      callback(*this);
    }

    if (steplength_ > th_stepdec_ && dV_ >= 0.) {
      decreaseRegularization(); // divides the preg_ by reg_decfactor_, which is 10 by default
    }
    if (steplength_ <= th_stepinc_ || std::abs(d_[1]) <= th_feas_) {
      if (preg_ == reg_max_) {
        STOP_PROFILER("SolverIntro::solve");
        std::cout << "returning from solve 1" << std::endl;
        exit(10);
        return false;
      }
      increaseRegularization();
    }

    if (is_feasible_ && stop_ < th_stop_) {
      STOP_PROFILER("SolverIntro::solve");
      std::cout << "returning from solve 2" << std::endl;
      exit(11);
      return true;
    }
  }
  STOP_PROFILER("SolverIntro::solve");
  std::cout << "returning from solve 3" << std::endl;
  return false;
}

double SolverIntro::tryStep(const double steplength) {
  forwardPass(steplength);
  hfeas_try_ = computeEqualityFeasibility();
  return cost_ - cost_try_;
}

double SolverIntro::stoppingCriteria() {
  stop_ = std::max(hfeas_, std::abs(d_[0] + 0.5 * d_[1]));
  return stop_;
}

void SolverIntro::resizeData() {
  START_PROFILER("SolverIntro::resizeData");
  SolverFDDP::resizeData();

  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    KQuu_tmp_[t].conservativeResize(ndx, nu);
    YZ_[t].conservativeResize(nu, nu);
    Hy_[t].conservativeResize(nh, nh);
    Qz_[t].conservativeResize(nh);
    Qzz_[t].conservativeResize(nh, nh);
    Qxz_[t].conservativeResize(ndx, nh);
    Quz_[t].conservativeResize(nu, nh);
    kz_[t].conservativeResize(nu);
    Kz_[t].conservativeResize(nu, ndx);
    ks_[t].conservativeResize(nh);
    Ks_[t].conservativeResize(nh, ndx);
    QuuinvHuT_[t].conservativeResize(nu, nh);
  }
  STOP_PROFILER("SolverIntro::resizeData");
}

double SolverIntro::calcDiff() {
  START_PROFILER("SolverIntro::calcDiff");
  SolverFDDP::calcDiff();
  const std::size_t T = problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  switch (eq_solver_) {
    case LuNull:
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
      for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
            models[t];
        const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = datas[t];
        if (model->get_nu() > 0 && model->get_nh() > 0) {
          Hu_lu_[t].compute(data->Hu);
          YZ_[t] << Hu_lu_[t].matrixLU().transpose(), Hu_lu_[t].kernel();
          Hu_rank_[t] = Hu_lu_[t].rank();
          const Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>
              Y = YZ_[t].leftCols(Hu_lu_[t].rank());
          Hy_[t].noalias() = data->Hu * Y;
          Hy_lu_[t].compute(Hy_[t]);
          const Eigen::Inverse<Eigen::PartialPivLU<Eigen::MatrixXd> > Hy_inv =
              Hy_lu_[t].inverse();
          ks_[t].noalias() = Hy_inv * data->h;
          Ks_[t].noalias() = Hy_inv * data->Hx;
          kz_[t].noalias() = Y * ks_[t];
          Kz_[t].noalias() = Y * Ks_[t];
        }
      }
      break;
    case QrNull:
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
      for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
            models[t];
        const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = datas[t];
        if (model->get_nu() > 0 && model->get_nh() > 0) {
          Hu_qr_[t].compute(data->Hu.transpose());
          YZ_[t] = Hu_qr_[t].householderQ();
          Hu_rank_[t] = Hu_qr_[t].rank();
          const Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>
              Y = YZ_[t].leftCols(Hu_qr_[t].rank());
          Hy_[t].noalias() = data->Hu * Y;
          Hy_lu_[t].compute(Hy_[t]);
          const Eigen::Inverse<Eigen::PartialPivLU<Eigen::MatrixXd> > Hy_inv =
              Hy_lu_[t].inverse();
          ks_[t].noalias() = Hy_inv * data->h;
          Ks_[t].noalias() = Hy_inv * data->Hx;
          kz_[t].noalias() = Y * ks_[t];
          Kz_[t].noalias() = Y * Ks_[t];
        }
      }
      break;
    case Schur:
      break;
  }

  STOP_PROFILER("SolverIntro::calcDiff");
  return cost_;
}

void SolverIntro::computeValueFunction(
    const std::size_t t, const boost::shared_ptr<ActionModelAbstract>& model) {
  const std::size_t nu = model->get_nu();
  Vx_[t] = Qx_[t];
  Vxx_[t] = Qxx_[t];
  if (nu != 0) {
    START_PROFILER("SolverIntro::Vx");
    Quuk_[t].noalias() = Quu_[t] * k_[t];
    Vx_[t].noalias() -= Qxu_[t] * k_[t];
    Qu_[t] -= Quuk_[t];
    Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    Qu_[t] += Quuk_[t];
    STOP_PROFILER("SolverIntro::Vx");
    START_PROFILER("SolverIntro::Vxx");
    KQuu_tmp_[t].noalias() = K_[t].transpose() * Quu_[t];
    KQuu_tmp_[t].noalias() -= 2 * Qxu_[t];
    Vxx_[t].noalias() += KQuu_tmp_[t] * K_[t];
    STOP_PROFILER("SolverIntro::Vxx");
  }
  Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
  Vxx_[t] = Vxx_tmp_;

  if (!std::isnan(preg_)) {
    Vxx_[t].diagonal().array() += preg_;
  }

  // Compute and store the Vx gradient at end of the interval (rollout state)
  if (!is_feasible_) {
    Vx_[t].noalias() += Vxx_[t] * fs_[t];
  }
}

void SolverIntro::computeGains(const std::size_t t) {
  START_PROFILER("SolverIntro::computeGains");
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      problem_->get_runningModels()[t];
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      problem_->get_runningDatas()[t];

  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  switch (eq_solver_) {
    case LuNull:
    case QrNull:
      if (nu > 0 && nh > 0) {
        START_PROFILER("SolverIntro::Qzz_inv");
        const std::size_t rank = Hu_rank_[t];
        const std::size_t nullity = data->Hu.cols() - rank;
        const Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>
            Z = YZ_[t].rightCols(nullity);
        Quz_[t].noalias() = Quu_[t] * Z;
        Qzz_[t].noalias() = Z.transpose() * Quz_[t];
        Qzz_llt_[t].compute(Qzz_[t]);
        STOP_PROFILER("SolverIntro::Qzz_inv");
        const Eigen::ComputationInfo& info = Qzz_llt_[t].info();
        if (info != Eigen::Success) {
          throw_pretty("backward error");
        }

        k_[t] = kz_[t];
        K_[t] = Kz_[t];
        Eigen::Transpose<Eigen::MatrixXd> QzzinvQzu = Quz_[t].transpose();
        Qzz_llt_[t].solveInPlace(QzzinvQzu);
        Qz_[t].noalias() = Z.transpose() * Qu_[t];
        Qzz_llt_[t].solveInPlace(Qz_[t]);
        Qxz_[t].noalias() = Qxu_[t] * Z;
        Eigen::Transpose<Eigen::MatrixXd> Qzx = Qxz_[t].transpose();
        Qzz_llt_[t].solveInPlace(Qzx);
        Qz_[t].noalias() -= QzzinvQzu * kz_[t];
        Qzx.noalias() -= QzzinvQzu * Kz_[t];
        k_[t].noalias() += Z * Qz_[t];
        K_[t].noalias() += Z * Qzx;
      } else {
        SolverFDDP::computeGains(t);
      }
      break;
    case Schur:
      SolverFDDP::computeGains(t);
      if (nu > 0 && nh > 0) {
        START_PROFILER("SolverIntro::Qzz_inv");
        QuuinvHuT_[t] = data->Hu.transpose();
        Quu_llt_[t].solveInPlace(QuuinvHuT_[t]);
        Qzz_[t].noalias() = data->Hu * QuuinvHuT_[t];
        Qzz_llt_[t].compute(Qzz_[t]);
        STOP_PROFILER("SolverIntro::Qzz_inv");
        const Eigen::ComputationInfo& info = Qzz_llt_[t].info();
        if (info != Eigen::Success) {
          throw_pretty("backward error");
        }
        Eigen::Transpose<Eigen::MatrixXd> HuQuuinv = QuuinvHuT_[t].transpose();
        Qzz_llt_[t].solveInPlace(HuQuuinv);
        ks_[t] = data->h;
        ks_[t].noalias() -= data->Hu * k_[t];
        Ks_[t] = data->Hx;
        Ks_[t].noalias() -= data->Hu * K_[t];
        k_[t].noalias() += QuuinvHuT_[t] * ks_[t];
        K_[t] += QuuinvHuT_[t] * Ks_[t];
      }
      break;
  }
  STOP_PROFILER("SolverIntro::computeGains");
}

EqualitySolverType SolverIntro::get_equality_solver() const {
  return eq_solver_;
}

double SolverIntro::get_th_feas() const { return th_feas_; }

double SolverIntro::get_rho() const { return rho_; }

double SolverIntro::get_upsilon() const { return upsilon_; }

bool SolverIntro::get_zero_upsilon() const { return zero_upsilon_; }

const std::vector<std::size_t>& SolverIntro::get_Hu_rank() const {
  return Hu_rank_;
}

const std::vector<Eigen::MatrixXd>& SolverIntro::get_YZ() const { return YZ_; }

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Qzz() const {
  return Qzz_;
}

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Qxz() const {
  return Qxz_;
}

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Quz() const {
  return Quz_;
}

const std::vector<Eigen::VectorXd>& SolverIntro::get_Qz() const { return Qz_; }

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Hy() const { return Hy_; }

const std::vector<Eigen::VectorXd>& SolverIntro::get_kz() const { return kz_; }

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Kz() const { return Kz_; }

const std::vector<Eigen::VectorXd>& SolverIntro::get_ks() const { return ks_; }

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Ks() const { return Ks_; }

void SolverIntro::set_equality_solver(const EqualitySolverType type) {
  eq_solver_ = type;
}

void SolverIntro::set_th_feas(const double th_feas) { th_feas_ = th_feas; }

void SolverIntro::set_rho(const double rho) {
  if (0. >= rho || rho > 1.) {
    throw_pretty("Invalid argument: "
                 << "rho value should between 0 and 1.");
  }
  rho_ = rho;
}

void SolverIntro::set_zero_upsilon(const bool zero_upsilon) {
  zero_upsilon_ = zero_upsilon;
}

}  // namespace crocoddyl
