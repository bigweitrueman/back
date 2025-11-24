# Robot Configuration Project

## Overview
This project is designed to facilitate the simulation and configuration of robotic systems using URDF (Unified Robot Description Format) files. It provides a structured approach to define robot parameters, parse URDF files, and simulate robot behavior.

## Project Structure
```
robot-config-project
├── src
│   ├── uni_sim.py            # Main script for the project, imports necessary libraries and defines URDF file path.
│   ├── robot_config.py       # Contains classes and methods for robot configuration, including loading URDF files and setting initial states.
│   ├── urdf_parser.py        # Logic for parsing URDF files, providing functions to read URDF content and extract joints, links, etc.
│   └── constants.py          # Defines constants used throughout the project, such as URDF file paths and default parameters.
├── config
│   ├── robot_params.yaml     # YAML configuration file containing robot parameters like joint limits, mass, inertia, etc.
│   └── simulation_config.yaml # YAML configuration file for simulation environment settings, such as timestep and gravity.
├── urdf
│   └── Lite3
│       └── Lite3_urdf       # Directory containing the new URDF file describing the robot's structure and characteristics.
├── tests
│   ├── test_robot_config.py  # Unit tests for functionalities in robot_config.py to ensure correct robot configuration.
│   └── test_urdf_parser.py   # Unit tests for functionalities in urdf_parser.py to ensure accurate URDF parsing.
├── requirements.txt          # Lists required Python libraries and their versions for the project.
└── README.md                 # Documentation file introducing the project, its purpose, usage, and installation steps.
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd robot-config-project
pip install -r requirements.txt
```

## Usage
1. Configure the robot parameters in `config/robot_params.yaml`.
2. Set up the simulation environment in `config/simulation_config.yaml`.
3. Run the main simulation script:

```bash
python src/uni_sim.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.