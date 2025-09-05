# X5 configuration in ` include/app/config.h `

```
┌─────────────────┐    CAN Bus    ┌─────────────────┐
│   Controller    │◄─────────────►│  Motor ID: 1    │ (Joint 1)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 2    │ (Joint 2)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 4    │ (Joint 3)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 5    │ (Joint 4)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 6    │ (Joint 5)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 7    │ (Joint 6)
│                 │               ├─────────────────┤
│                 │◄─────────────►│  Motor ID: 8    │ (Gripper)
└─────────────────┘               └─────────────────┘
```

```cpp
// joint_names: [0: joint1, 1: joint2, 2: joint3, 3: joint4, 4: joint5, 5: joint6]
// motors: [0: EC_A4310, 1: EC_A4310, 2: EC_A4310, 3: DM_J4310, 4: DM_J4310, 5: DM_J4310]
RobotConfigFactory()
    {
        configurations["X5"] = std::make_shared<RobotConfig>(
            "X5",                                                          // robot_model
            (VecDoF(6) << -3.14, -0.05, -0.1, -1.6, -1.57, -2).finished(), // joint_pos_min
            (VecDoF(6) << 2.618, 3.50, 3.20, 1.55, 1.57, 2).finished(),    // joint_pos_max
            (VecDoF(6) << 5.0, 5.0, 5.5, 5.5, 5.0, 5.0).finished(),        // joint_vel_max
            (VecDoF(6) << 30.0, 40.0, 30.0, 15.0, 10.0, 10.0).finished(),  // joint_torque_max
            (Pose6d() << 0.6, 0.6, 0.6, 1.8, 1.8, 1.8).finished(),         // ee_vel_max
            0.3,                                                           // gripper_vel_max
            1.5,                                                           // gripper_torque_max
            0.088,                                                         // gripper_width
            5.03,                                                          // gripper_open_readout
            6,                                                             // joint_dof
            std::vector<int>{1, 2, 4, 5, 6, 7},                            // motor_id
            std::vector<MotorType>{MotorType::EC_A4310, MotorType::EC_A4310, MotorType::EC_A4310, MotorType::DM_J4310,
                                   MotorType::DM_J4310, MotorType::DM_J4310}, // motor_type
            8,                                                                // gripper_motor_id
            MotorType::DM_J4310,                                              // gripper_motor_type
            (Eigen::Vector3d() << 0, 0, -9.807).finished(),                   // gravity_vector
            "base_link",                                                      // base_link_name
            "eef_link",                                                       // eef_link_name
            std::string(SDK_ROOT) + "/models/X5.urdf"                         // urdf_path
        );
    }
```
