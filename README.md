# Agression Lunging Simulation

Please copy the preprogrammed.py and fly.py to update the class definition and restart the kernel to ensure the program runs correctly.
Please run the flies.py class definition to enable the simulation for the "overall performance".

# Lunging and Force Measurement File ###
incorporates the following 

- Lunging behavior simulation manipulation data

- Multi-camera rendering of one-fly simulation

- Multi-fly simulation (target stationary)

- Force measurement plot

- Plot of the simulation result



Running the first four code blocks would be able to provide the basic setup for the lunging behavior 

The 5th block would act like a sanity check to verify the actuation angle

The Multi-Camera rendering and Multi-fly simulation could be run to produce videos of lunging.

All the above code could be run simultaneously.

The Rendering blocks (Multi-fly Simulation) and the Peak Force measurement blocks could not be implemented in one run. 

- To perform peak force measurement, please restart the kernel, uncomment the measurement block, and comment out the rendering blocks
    
- To switch back to rendering, undo the above process.

    
The last two blocks are plots of measurement data from the manipulation of Peak force measurement.

# Overall Performance File ###
This file incorporates the above lunging flies.
The lunging fly performing distance detection, chasing and lunging
Simulating the fly chasing and lunging dynamically.


- Removement of parts of the fly's body that are in its field of vision

- Manually add desired collison pairs using the function. It can be freely adjusted

- Plot of the average weighted pixel value over time 

Overall Performance.ipynb Usage Guide
This Jupyter notebook contains key steps for simulating detection, transitioning, and aggressive lunging behaviors using Python. 

Code Block 1: Basic Setup
Imports necessary libraries 
Loads simulation and modeling components like Simulation, Camera, and FlatTerrain.
Executing this block sets up the environment for the simulations.

Code Block 2: Import Fly Models
Imports the LungingFly and WalkingFly models from the flies module.
This block is essential for initializing the models used in the simulations.

Code Block 3: Configure Vision
Modifies the vision configurations for the WalkingFly to hide certain body segments from its field of vision.
This step is crucial for accurately simulating the visual field of the fly model.

Code Block 4: Collision Pairs and Functions Definitions
Defines functions to add collision pairs between different parts of two fly models.
Adds specific collision pairs that are necessary for simulating interactions during the lunging behavior.

Code Block 5: Initialize Simulation
Sets the simulation time parameters.
Initializes two fly models with specific properties and positions.
Enables collision detection and sets up a flat terrain arena.
Adds cameras to the simulation for recording the scenario.
This block is pivotal for starting the simulation with the configured parameters and models.

Code Block 5: Plot the average weighted pixel value over time

Code Block 6: Play the video

