    ```
    brain2bot/
    |-- CMakeLists.txt
    |-- package.xml  
    |-- src/                  # Python ROS nodes
    |   |-- acquisition.py  
    |   |-- feature_extraction.py  
    |   |-- classification.py
    |   |-- control.py
    |   |-- utils/            # Utility scripts/modules
    |       |-- streamers.py    # OfflineFileStreamer / OnlineRDAStreamer
    |       |-- extractors.py   # Bandpower / CSP
    |       |-- classifiers.py  # ML / DL
    |       |-- simulators.py   # Unity / HabitatSim / Gazebo
    |       |-- robot.py        #TODO: KinovaJacoArm
    |-- launch/  
        |-- calibration.launch  
        |-- onlineML.launch  
        |-- onlineDL.launch  
        |-- pseudo-online.launch  
    ```
