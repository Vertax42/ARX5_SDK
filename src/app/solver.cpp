#include "app/solver.h"

#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/utilities/utility.h>
#include <kdl_parser/kdl_parser.hpp>

#include <stdexcept>
#include <string>
#include <cmath>
#include <random>
#include <cstdio>

namespace arx
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static KDL::Chain strip_fixed_joints(const KDL::Chain &chain)
{
    KDL::Chain result;
    for (unsigned int i = 0; i < chain.getNrOfSegments(); ++i) {
        const KDL::Segment &seg = chain.getSegment(i);
        if (seg.getJoint().getType() != KDL::Joint::None)
            result.addSegment(seg);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Constructor: minimal (no base/eef/gravity override)
// ---------------------------------------------------------------------------
Arx5Solver::Arx5Solver(std::string urdf_path, int joint_dof,
                        Eigen::VectorXd joint_pos_min,
                        Eigen::VectorXd joint_pos_max)
    : Arx5Solver(std::move(urdf_path), joint_dof,
                 std::move(joint_pos_min), std::move(joint_pos_max),
                 "base_link", "eef_link",
                 Eigen::Vector3d(0.0, 0.0, -9.807))
{}

// ---------------------------------------------------------------------------
// Constructor: full (with base_link, eef_link, gravity_vector)
// ---------------------------------------------------------------------------
Arx5Solver::Arx5Solver(std::string urdf_path, int joint_dof,
                        Eigen::VectorXd joint_pos_min,
                        Eigen::VectorXd joint_pos_max,
                        std::string base_link, std::string eef_link,
                        Eigen::Vector3d gravity_vector)
    : JOINT_DOF_(joint_dof),
      JOINT_POS_MIN_(std::move(joint_pos_min)),
      JOINT_POS_MAX_(std::move(joint_pos_max))
{
    if (JOINT_POS_MIN_.size() != (Eigen::Index)joint_dof ||
        JOINT_POS_MAX_.size() != (Eigen::Index)joint_dof) {
        throw std::invalid_argument(
            "joint_pos_min/max size (" +
            std::to_string(JOINT_POS_MIN_.size()) + "/" +
            std::to_string(JOINT_POS_MAX_.size()) +
            ") does not match joint_dof (" + std::to_string(joint_dof) + ")");
    }

    if (!kdl_parser::treeFromFile(urdf_path, tree_)) {
        throw std::runtime_error("Failed to load URDF from: " + urdf_path);
    }

    // FK/IK chain: base_link → eef_link
    if (!tree_.getChain(base_link, eef_link, chain_)) {
        throw std::runtime_error(
            "Failed to extract KDL chain from " + base_link + " to " + eef_link);
    }

    // chain_without_fixed_joints_: eef_link chain with fixed segments removed
    // (= base_link → link6, used for inverse dynamics)
    chain_without_fixed_joints_ = strip_fixed_joints(chain_);

    if ((int)chain_without_fixed_joints_.getNrOfJoints() != joint_dof) {
        throw std::invalid_argument(
            "URDF chain has " +
            std::to_string(chain_without_fixed_joints_.getNrOfJoints()) +
            " DOF but joint_dof=" + std::to_string(joint_dof));
    }

    printf("Parsed %d non-fixed joints and %d segments. Base link: %s. "
           "Using %s for kinematics and %s for inverse dynamics.\n",
           (int)chain_without_fixed_joints_.getNrOfJoints(),
           (int)chain_.getNrOfSegments(),
           base_link.c_str(),
           eef_link.c_str(),
           "link6");

    KDL::Vector grav(gravity_vector(0), gravity_vector(1), gravity_vector(2));

    ik_solver_ = std::make_shared<KDL::ChainIkSolverPos_LMA>(
        chain_, EPS_, MAXITER_, EPS_JOINTS_);  // full chain incl. fixed eef segment
    fk_solver_ = std::make_shared<KDL::ChainFkSolverPos_recursive>(
        chain_);  // full chain including fixed eef segment
    id_solver_ = std::make_shared<KDL::ChainIdSolver_RNE>(
        chain_without_fixed_joints_, grav);
}

// ---------------------------------------------------------------------------
// Forward kinematics
// ---------------------------------------------------------------------------
Eigen::Matrix<double, 6, 1> Arx5Solver::forward_kinematics(Eigen::VectorXd joint_pos)
{
    if (joint_pos.size() != JOINT_DOF_) {
        throw std::invalid_argument(
            "joint_pos size " + std::to_string(joint_pos.size()) +
            " != JOINT_DOF_ " + std::to_string(JOINT_DOF_));
    }

    KDL::JntArray q(JOINT_DOF_);
    for (int i = 0; i < JOINT_DOF_; ++i)
        q(i) = joint_pos(i);

    KDL::Frame frame;
    int ret = fk_solver_->JntToCart(q, frame);
    if (ret < 0) {
        throw std::runtime_error(
            "FK solver failed with error: " + std::to_string(ret));
    }

    // Extract position
    double x = frame.p.x();
    double y = frame.p.y();
    double z = frame.p.z();

    // Extract RPY from rotation matrix
    double roll, pitch, yaw;
    frame.M.GetRPY(roll, pitch, yaw);

    Eigen::Matrix<double, 6, 1> pose;
    pose << x, y, z, roll, pitch, yaw;
    return pose;
}

// ---------------------------------------------------------------------------
// Inverse kinematics (single attempt)
// ---------------------------------------------------------------------------
std::tuple<int, Eigen::VectorXd>
Arx5Solver::inverse_kinematics(Eigen::Matrix<double, 6, 1> target_pose_6d,
                                Eigen::VectorXd current_joint_pos)
{
    if (current_joint_pos.size() != JOINT_DOF_) {
        throw std::invalid_argument(
            "current_joint_pos size " + std::to_string(current_joint_pos.size()) +
            " != JOINT_DOF_ " + std::to_string(JOINT_DOF_));
    }

    // Build target frame from x,y,z,roll,pitch,yaw
    KDL::Frame target;
    target.p = KDL::Vector(target_pose_6d(0), target_pose_6d(1), target_pose_6d(2));
    target.M = KDL::Rotation::RPY(target_pose_6d(3), target_pose_6d(4), target_pose_6d(5));

    KDL::JntArray q_init(JOINT_DOF_);
    for (int i = 0; i < JOINT_DOF_; ++i)
        q_init(i) = current_joint_pos(i);

    KDL::JntArray q_out(JOINT_DOF_);
    int ik_status = ik_solver_->CartToJnt(q_init, target, q_out);

    Eigen::VectorXd result(JOINT_DOF_);
    for (int i = 0; i < JOINT_DOF_; ++i)
        result(i) = q_out(i);

    return {ik_status, result};
}

// ---------------------------------------------------------------------------
// Multi-trial inverse kinematics
// ---------------------------------------------------------------------------
std::tuple<int, Eigen::VectorXd>
Arx5Solver::multi_trial_ik(Eigen::Matrix<double, 6, 1> target_pose_6d,
                            Eigen::VectorXd current_joint_pos,
                            int additional_trial_num)
{
    // First attempt from current joint position
    auto [status, result] = inverse_kinematics(target_pose_6d, current_joint_pos);
    if (status == KDL::SolverI::E_NOERROR)
        return {status, result};

    // Additional random-seeded trials
    std::mt19937 rng(std::random_device{}());
    for (int trial = 0; trial < additional_trial_num; ++trial) {
        Eigen::VectorXd seed(JOINT_DOF_);
        for (int i = 0; i < JOINT_DOF_; ++i) {
            std::uniform_real_distribution<double> dist(
                JOINT_POS_MIN_(i), JOINT_POS_MAX_(i));
            seed(i) = dist(rng);
        }
        auto [s2, r2] = inverse_kinematics(target_pose_6d, seed);
        if (s2 == KDL::SolverI::E_NOERROR)
            return {s2, r2};
        // Keep this result if current best hasn't converged
        if (status != KDL::SolverI::E_NOERROR) {
            status = s2;
            result = r2;
        }
    }

    return {status, result};
}

// ---------------------------------------------------------------------------
// Inverse dynamics (recursive Newton-Euler)
// ---------------------------------------------------------------------------
Eigen::VectorXd Arx5Solver::inverse_dynamics(Eigen::VectorXd joint_pos,
                                              Eigen::VectorXd joint_vel,
                                              Eigen::VectorXd joint_acc)
{
    if (joint_pos.size() != JOINT_DOF_ ||
        joint_vel.size() != JOINT_DOF_ ||
        joint_acc.size() != JOINT_DOF_) {
        throw std::invalid_argument(
            "inverse_dynamics: input vector size mismatch (expected " +
            std::to_string(JOINT_DOF_) + ")");
    }

    KDL::JntArray q(JOINT_DOF_), qd(JOINT_DOF_), qdd(JOINT_DOF_);
    for (int i = 0; i < JOINT_DOF_; ++i) {
        q(i)   = joint_pos(i);
        qd(i)  = joint_vel(i);
        qdd(i) = joint_acc(i);
    }

    // External wrenches (all zero)
    KDL::Wrenches f_ext(chain_without_fixed_joints_.getNrOfSegments());

    KDL::JntArray torques(JOINT_DOF_);
    int ret = id_solver_->CartToJnt(q, qd, qdd, f_ext, torques);
    if (ret < 0) {
        throw std::runtime_error(
            "ID solver failed with error: " + std::to_string(ret));
    }

    Eigen::VectorXd result(JOINT_DOF_);
    for (int i = 0; i < JOINT_DOF_; ++i)
        result(i) = torques(i);
    return result;
}

// ---------------------------------------------------------------------------
// IK status name lookup
// ---------------------------------------------------------------------------
std::string Arx5Solver::get_ik_status_name(int ik_status)
{
    auto it = IK_STATUS_MAP_.find(ik_status);
    if (it != IK_STATUS_MAP_.end())
        return it->second;
    return "Unknown IK status";
}

} // namespace arx
