// test_solver.cpp
// Tests Arx5Solver FK / IK / ID with the X5 URDF.
// Produces deterministic numeric output; run against both lib versions and diff.

#include "app/solver.h"
#include <cstdio>
#include <cmath>
#include <Eigen/Core>

static int g_pass = 0, g_fail = 0;

static void check(const char *name, double got, double expected, double tol = 1e-4)
{
    double err = std::fabs(got - expected);
    if (err <= tol) {
        printf("PASS %-50s got=%+.8f\n", name, got);
        g_pass++;
    } else {
        printf("FAIL %-50s got=%+.8f  expected=%+.8f  err=%.2e\n",
               name, got, expected, err);
        g_fail++;
    }
}

static void print_vec(const char *label, const Eigen::VectorXd &v)
{
    printf("%-20s:", label);
    for (int i = 0; i < v.size(); i++) printf(" %+.6f", v(i));
    printf("\n");
}

static void print_pose(const char *label, const Eigen::Matrix<double,6,1> &p)
{
    printf("%-20s: x=%+.6f y=%+.6f z=%+.6f roll=%+.6f pitch=%+.6f yaw=%+.6f\n",
           label, p(0),p(1),p(2),p(3),p(4),p(5));
}

int main(int argc, char **argv)
{
    const char *urdf = (argc > 1) ? argv[1]
                                  : "/home/vertax/ARX5_SDK/models/X5.urdf";
    const int DOF = 6;

    Eigen::VectorXd qmin(DOF), qmax(DOF);
    // typical ARX5 joint limits (radians)
    qmin << -3.14, -2.96, -0.087, -3.14, -1.92, -3.14;
    qmax <<  3.14,  2.96,  3.14,   3.14,  1.92,  3.14;

    arx::Arx5Solver solver(urdf, DOF, qmin, qmax);

    printf("=== Arx5Solver tests (URDF: %s) ===\n\n", urdf);

    // -------------------------------------------------------------------------
    // Test 1: FK at zero position
    // -------------------------------------------------------------------------
    {
        Eigen::VectorXd q = Eigen::VectorXd::Zero(DOF);
        auto pose = solver.forward_kinematics(q);
        print_pose("FK(zeros)", pose);

        // Save as reference for IK round-trip
        printf("\n");
    }

    // -------------------------------------------------------------------------
    // Test 2: FK at several known joint configs
    // -------------------------------------------------------------------------
    struct QTest { const char *name; double q[6]; } qtests[] = {
        {"FK_q1", { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0}},
        {"FK_q2", { 0.5, -0.3,  0.7, -0.5,  0.3, -0.2}},
        {"FK_q3", {-1.0,  1.0, -1.0,  1.0, -1.0,  1.0}},
        {"FK_q4", { 1.57, 0.0,  1.57, 0.0,  1.0,  0.0}},
        {"FK_q5", {-0.5, -0.5, -0.5, -0.5, -0.5, -0.5}},
    };
    for (auto &t : qtests) {
        Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(t.q, DOF);
        auto pose = solver.forward_kinematics(q);
        print_pose(t.name, pose);
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // Test 3: IK → FK round-trip (IK of FK result should give same pose)
    // -------------------------------------------------------------------------
    printf("=== IK → FK round-trip ===\n");
    struct RoundTripTest { const char *name; double q[6]; } rttests[] = {
        {"RT_zeros", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
        {"RT_mid",   {0.5,-0.3, 0.7,-0.5, 0.3,-0.2}},
        {"RT_lims",  {1.0, 1.0,-1.0, 1.0,-1.0, 1.0}},
    };
    for (auto &t : rttests) {
        Eigen::VectorXd q_ref = Eigen::Map<const Eigen::VectorXd>(t.q, DOF);
        auto pose_ref = solver.forward_kinematics(q_ref);

        auto [status, q_ik] = solver.inverse_kinematics(pose_ref, q_ref);
        auto pose_ik = solver.forward_kinematics(q_ik);

        // Compare FK pose from IK solution vs original pose
        double pos_err = (pose_ik.head<3>() - pose_ref.head<3>()).norm();
        double rot_err = (pose_ik.tail<3>() - pose_ref.tail<3>()).norm();

        char bpos[80], brot[80];
        snprintf(bpos,sizeof(bpos),"%s pos_err(m)",t.name);
        snprintf(brot,sizeof(brot),"%s rot_err(rad)",t.name);
        printf("  IK status: %d (%s)\n", status,
               solver.get_ik_status_name(status).c_str());
        print_vec("  q_ik", q_ik);
        check(bpos, pos_err, 0.0, 1e-3);
        check(brot, rot_err, 0.0, 1e-2);
        printf("\n");
    }

    // -------------------------------------------------------------------------
    // Test 4: multi_trial_ik
    // -------------------------------------------------------------------------
    printf("=== multi_trial_ik ===\n");
    {
        Eigen::VectorXd q0(DOF);
        q0 << 0.3, -0.5, 0.8, -0.3, 0.5, -0.1;
        auto target_pose = solver.forward_kinematics(q0);
        Eigen::VectorXd q_init = Eigen::VectorXd::Zero(DOF);

        auto [status, q_out] = solver.multi_trial_ik(target_pose, q_init, 10);
        auto pose_out = solver.forward_kinematics(q_out);
        double pos_err = (pose_out.head<3>() - target_pose.head<3>()).norm();

        printf("  multi_trial_ik status: %d (%s)\n", status,
               solver.get_ik_status_name(status).c_str());
        print_vec("  q_out", q_out);
        check("multi_trial_ik pos_err(m)", pos_err, 0.0, 5e-3);
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // Test 5: inverse_dynamics (gravity compensation at zero vel/acc)
    // -------------------------------------------------------------------------
    printf("=== inverse_dynamics ===\n");
    struct IDTest { const char *name; double q[6]; } idtests[] = {
        {"ID_zeros", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
        {"ID_q1",    {0.5,-0.3, 0.7,-0.5, 0.3,-0.2}},
        {"ID_q2",    {0.0, 1.57, 0.0, 0.0, 0.0, 0.0}},
    };
    for (auto &t : idtests) {
        Eigen::VectorXd q   = Eigen::Map<const Eigen::VectorXd>(t.q, DOF);
        Eigen::VectorXd qd  = Eigen::VectorXd::Zero(DOF);
        Eigen::VectorXd qdd = Eigen::VectorXd::Zero(DOF);
        auto tau = solver.inverse_dynamics(q, qd, qdd);
        print_vec(t.name, tau);
    }
    printf("\n");

    // -------------------------------------------------------------------------
    // Test 6: get_ik_status_name
    // -------------------------------------------------------------------------
    printf("=== get_ik_status_name ===\n");
    int codes[] = {0, -1, -2, -3, -4, -5, -6, -7, -8, -9, 99};
    for (int c : codes)
        printf("  status[%3d] = \"%s\"\n", c,
               solver.get_ik_status_name(c).c_str());
    printf("\n");

    printf("=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
