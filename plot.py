

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from robot import HexaRobot
import time
from gait_planner import MoveGaitPlanner, GAIT

def plot_3d_line():
    """
    Plot a 3D line from a list of 3D coordinates
    
    Args:
        coordinates: List of tuples/lists in format [(x1, y1, z1), (x2, y2, z2), ...]
    """

    plt.ion()
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add title
    ax.set_title('3D Line Plot')
    ax.set_box_aspect([1,1,1])
    lines_objs = []

    robot = HexaRobot()

    gait = MoveGaitPlanner(robot)


    for coordinates in robot.get_plot_links():
        # Convert to numpy array for easier manipulation
        coords = np.array(coordinates)
        
        # Extract x, y, z coordinates
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
    
        # Plot the line
        lines_objs.append(ax.plot(x, y, z, marker='o', linewidth=2, markersize=6))

    
    current_time = time.time()
    while True:
        dt = time.time() - current_time
        gait.step(dt, (0.1, 0))
        current_time = current_time + dt
        for i, coordinates in enumerate(robot.get_plot_links()):
            # Convert to numpy array for easier manipulation
            coords = np.array(coordinates)
            
            # Extract x, y, z coordinates
            x = coords[:, 0]
            y = coords[:, 1]
            z = coords[:, 2]
            
            # Plot the line
            lines_objs[i][0].set_data_3d(x, y, z)
        plt.pause(0.001)
        
        
    plt.ioff()
    plt.show()
        




plot_3d_line()