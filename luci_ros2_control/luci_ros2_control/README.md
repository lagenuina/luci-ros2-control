### Connect to LUCI
ros2 run luci_grpc_interface grpc_interface_node -a 10.2.10.3

### Launch RTAB-Map
ros2 launch sar_risk_maps dual_camera_rtab.launch.py localization:=false

### Launch local planner
ros2 launch luci_ros2_control luci_ros2_control.launch.py

## Important Note
`twist_to_luci_joystick.py` converts velocity commands published on `/cmd_vel` into joystick deflection commands for LUCI.

`luci_speed_profiles.ods` contains joystick deflection (10-100) mapped to the resulting linear speed (mph) and angular speed (mph), measured separately for each of LUCI's 5 speed profiles. For linear velocity, an inverse cubic (`deflection = a*v^3 + b*v^2 + c*v`) was fit per profile to this data — these fits are the `PROFILE_CUBICS` coefficients in the script.

The equivalent per-profile fit for angular velocity has *not* been wired in yet: the script uses a single hardcoded linear conversion that was provided by LUCI.

**When running the planner, make sure LUCI is set to Profile 5.**