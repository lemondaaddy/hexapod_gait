import numpy as np

def jacobian_inverse_leg_robot(theta1, theta2, theta3, l1, l2, l3, force_tip):
    """
    Calculate joint torques from tip forces using Jacobian matrix
    for a 3-DOF leg-type robotic arm.
    Joint 1: Rotation around Z-axis (hip joint)
    Joint 2 & 3: Rotation around X-axis (knee and ankle joints)
    
    Parameters:
    - theta1: Hip joint angle (rotation around Z-axis) in radians
    - theta2: Knee joint angle (rotation around X-axis) in radians  
    - theta3: Ankle joint angle (rotation around X-axis) in radians
    - l1: Length from hip to knee (first link length)
    - l2: Length from knee to ankle (second link length)
    - l3: Length from ankle to foot tip (third link length)
    - force_tip: Force vector at the end-effector [fx, fy, fz]
    
    Returns:
    - torques: Joint torques [tau1, tau2, tau3]
    """
    
    # Calculate trigonometric values
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c3 = np.cos(theta3)
    s3 = np.sin(theta3)
    
    # Calculate position of end effector
    # Using a typical leg configuration where links extend along local Y-axes initially
    x = l1 * s1 + c1 * (l2 * s2 + l3 * (s2 * c3 + c2 * s3))
    y = -l1 * c1 + s1 * (l2 * s2 + l3 * (s2 * c3 + c2 * s3))
    z = l2 * c2 + l3 * (c2 * c3 - s2 * s3)
    
    # Calculate Jacobian matrix
    # J = [[dx/dtheta1, dx/dtheta2, dx/dtheta3],
    #      [dy/dtheta1, dy/dtheta2, dy/dtheta3],
    #      [dz/dtheta1, dz/dtheta2, dz/dtheta3]]
    
    # Partial derivatives for the leg robot configuration
    dx_dtheta1 = l1 * c1 + (l2 * s2 + l3 * (s2 * c3 + c2 * s3)) * c1 - (l2 * c2 + l3 * (c2 * c3 - s2 * s3)) * s1
    dx_dtheta2 = c1 * (l2 * c2 + l3 * (c2 * c3 - s2 * s3))
    dx_dtheta3 = c1 * l3 * (c2 * c3 - s2 * s3)
    
    dy_dtheta1 = l1 * s1 + (l2 * s2 + l3 * (s2 * c3 + c2 * s3)) * s1 + (l2 * c2 + l3 * (c2 * c3 - s2 * s3)) * c1
    dy_dtheta2 = s1 * (l2 * c2 + l3 * (c2 * c3 - s2 * s3))
    dy_dtheta3 = s1 * l3 * (c2 * c3 - s2 * s3)
    
    dz_dtheta1 = 0
    dz_dtheta2 = -(l2 * s2 + l3 * (s2 * c3 + c2 * s3))
    dz_dtheta3 = -l3 * (s2 * c3 + c2 * s3)
    
    # Construct Jacobian matrix
    jacobian = np.array([
        [dx_dtheta1, dx_dtheta2, dx_dtheta3],
        [dy_dtheta1, dy_dtheta2, dy_dtheta3],
        [dz_dtheta1, dz_dtheta2, dz_dtheta3]
    ])
    
    # Check if Jacobian is singular (determinant close to zero)
    det_j = np.linalg.det(jacobian)
    if abs(det_j) < 1e-8:
        print("Warning: Jacobian matrix is near-singular")
    
    # Convert force at tip to joint torques: tau = J^T * F
    # For static equilibrium, joint torques are related to tip forces by transpose of Jacobian
    force_tip = np.array(force_tip).reshape(3, 1)
    torques = np.dot(jacobian.T, force_tip).flatten()
    
    return torques, jacobian

def jacobian_inverse_leg_robot_simple(theta1, theta2, theta3, l1, l2, l3, force_x, force_y, force_z):
    """
    Simplified calculation of joint torques from forces for leg-type robotic arm
    Joint 1: Rotation around Z-axis (hip joint)
    Joint 2 & 3: Rotation around X-axis (knee and ankle joints)
    
    Parameters:
    - theta1: Hip joint angle (rotation around Z-axis) in radians
    - theta2: Knee joint angle (rotation around X-axis) in radians  
    - theta3: Ankle joint angle (rotation around X-axis) in radians
    - l1: Length from hip to knee
    - l2: Length from knee to ankle
    - l3: Length from ankle to foot tip
    - force_x, force_y, force_z: Forces at the end-effector in x, y, z directions
    
    Returns:
    - torques: Joint torques [tau1, tau2, tau3]
    """
    
    # Calculate trigonometric values
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c3 = np.cos(theta3)
    s3 = np.sin(theta3)
    
    # Calculate the elements of the Jacobian matrix based on the leg robot kinematics
    # For a leg with hip rotating about Z and knees about X:
    
    # dx/dt1, dx/dt2, dx/dt3
    dx_dt1 = c1 * (l2 * s2 + l3 * (s2 * c3 + c2 * s3)) - s1 * (l2 * c2 + l3 * (c2 * c3 - s2 * s3))
    dx_dt2 = c1 * (l2 * c2 + l3 * (c2 * c3 - s2 * s3))
    dx_dt3 = c1 * l3 * (c2 * c3 - s2 * s3)
    
    # dy/dt1, dy/dt2, dy/dt3
    dy_dt1 = s1 * (l2 * s2 + l3 * (s2 * c3 + c2 * s3)) + c1 * (l2 * c2 + l3 * (c2 * c3 - s2 * s3))
    dy_dt2 = s1 * (l2 * c2 + l3 * (c2 * c3 - s2 * s3))
    dy_dt3 = s1 * l3 * (c2 * c3 - s2 * s3)
    
    # dz/dt1, dz/dt2, dz/dt3
    dz_dt1 = 0
    dz_dt2 = -(l2 * s2 + l3 * (s2 * c3 + c2 * s3))
    dz_dt3 = -l3 * (s2 * c3 + c2 * s3)
    
    # Build the Jacobian matrix
    jacobian = np.array([
        [dx_dt1, dx_dt2, dx_dt3],
        [dy_dt1, dy_dt2, dy_dt3],
        [dz_dt1, dz_dt2, dz_dt3]
    ])
    
    # Calculate torques using transpose of Jacobian: tau = J^T * F
    forces = np.array([force_x, force_y, force_z])
    torques = np.dot(jacobian.T, forces)
    
    return torques, jacobian




def jacobian(theta1, theta2, theta3, l1, l2, l3):
    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)

    s23 = np.sin(theta2 + theta3)
    c23 = np.cos(theta2 + theta3)

    J = np.array([
        [-s1* (l2 * s2 + l3 * s23), c1 * (l2 * c2 + l3 * c23), c1 * l3 * c23],
        [c1 * (l2 * s2 + l3 * s23) , s1 * (l2 * c2 + l3 * c23), s1 * l3 * c23],
        [0, l2 * s2 + l3 * s23, l3 * s23]
    ])

    return J


"""
d_s1 = c1 * d_theta1
d_c1 = -s1 * d_theta1
d_s23 = c23 * d_theta2 + c23 * d_theta3
d_c23 = -s23 * d_theta2 - s23 * d_theta3
"""
def jacobian_simple( theta1, theta2, theta3, l1, l2, l3):
    d_theta1 = 0
    d_theta2 = 0
    d_theta3 = 0

    s1 = np.sin(theta1)
    s2 = np.sin(theta2)
    c1 = np.cos(theta1)
    c2 = np.cos(theta2)

    s23 = np.sin(theta2 + theta3)
    c23 = np.cos(theta2 + theta3)



    """
    x = 0
    y = l2 * c2  + l3 * c23
    z = -l2* s2 - l3 * s23


    dy = -l2 * s2 * d2 - l3 * s23 * d2 - l3 * s23 *d3
    dz = -l2 * c2 * d2 - l3 * c23 * d2 - l3 * c23 * d3
    """
    J = np.array([
         [-l2 * s2 - l3 * s23, -l3 * s23],
         [ -l2 * c2 - l3 * c23, -l3 * c23],
       
        ])

    """
    x = -l1 * s1 - l2 * s1 * c2 - l3 * s1 * c23
    y =  l1 * c1 + l2 * c1 * c2 + l3 * c1 * c23


    dy = -l1 * s1 * d1 - l2 * s1 * c2 *d1 - l2 * c1 * s2 * d2 - l3 * s1 * c23 *d1 - l3 * c1 * s23 * ( d2 + d3)

     (-l1 * s1 - l2 * s1 *c2 -l3 * s1 * c23) 
    - s1 *( l1 + l2 * c2 + l3 * c23)

    -l2 * c1 * s2 - l3 * c1 * s23 = -c1*(l2 *s2 + l3 * s23)

    z =  -l2 * s2 - l3 * s23 = l2 * c2 * d2 + l3 * c23 * d2 + l3 * c23 * d3

    
    y =  l1 * c1 + l2 * c1 * c2 + l3 * c1 * c23
    x =  l1 * s1 + l2 * s1 * c2 + l3 * s1 * c23
    dx = l1 * c1 * d_theta1 + l2 * c1 * c2 * d_theta1 - l2 * s1 * s2 * d_theta2 + l3 * c1 * c23 * d_theta1 - l3 * s1 * s23 * d_theta2 - l3 * s1 * s23 * d_theta3 

    dx = -(l1 * c1 + l2 * c1 * c2 + l3 *c1 * c23) 
        - c1 * ( l1 + l2 * c2 + l3 * c23)

        s1 (  l2  * s2 + l3 * s23  )
   

    J =np.array( [
            [  -c1 * ( l1 + l2 * c2 + l3 * c23),  s1 * ( l2 * s2 + l3 * s23),  l3 * s1 * s23],
            [  -s1 * ( l1 + l2 * c2 + l3 * c23), -c1 * ( l2 * s2 + l3 * s23), -l3 * c1 * s23],
            [  0,                                 -(l2 * c2 + l3 * c23),           -l3 * c23],
            
         ])
     """
    X = l1 
    Y = -l2 * s1 - l3 * s23
    Z = l2 * c2 + l3 * c23
    return np.array([
        [ -(X * s1  + Y * c1), Z * s1, l3 * c23 * s1], 
        [ X * c1 - Y * s1, -Z * c1, -l3 * c23 * c1],
        [0, Y, -l3 * s23]
])