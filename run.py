import mujoco
import mujoco.viewer
import numpy as np
import time
from model import Body
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

def quaternion_to_euler_scipy(q, sequence='zyx'):

    # 创建 Rotation 对象
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # scipy 使用 [x, y, z, w]
    
    # 转换为欧拉角
    euler = rot.as_euler(sequence, degrees=False)
    
    return euler
    



def get_all_joint_torques(model, data):
    """获取所有关节的力矩"""
    torques = {}
    for actuator_idx in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_idx)
        joint_id = model.actuator_trnid[actuator_idx, 0]
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        torque = f'{ data.qfrc_actuator[actuator_idx + 6]:.2f} Nm'
        torques[joint_name] = torque
    return torques


model = mujoco.MjModel.from_xml_path('./robot/urdf/robot.mjcf')
data = mujoco.MjData(model)
model.opt.timestep = 0.002

mode = 1
radius = 0.18#0.20#0.178
height = -0.045010717660720456

#radius = 0.178#0.20#0.178
#height = -0.045010717660720456

#radius = 0.22
#height = -0.045010717660720456


#radius = 0.178
#height = -0.08

body = Body(height, radius, x_offset = -0.01, z_offset = 0.02)

target_angles = []
   
target_angles.extend([body.arms[0].A0,  0, body.arms[0].A1 , body.arms[0].A2, 0])
for i in range(1, 3):
    target_angles.extend([body.arms[i].A0,  body.arms[i].A1, body.arms[i].A2])

for i in range(3, 5):
    target_angles.extend([body.arms[i].A0,  -body.arms[i].A1, -body.arms[i].A2])

target_angles.extend([body.arms[5].A0,  0, -body.arms[5].A1, -body.arms[5].A2, 0])
#qpos = np.zeros(model.nq)
#qpos[7:] = target_angles
data.qpos[0:7] = [0,0,0.078,0,0,0,0]
#data.xpos[0:7][1] = [0,0,10]
data.qpos[7:] = target_angles

mujoco.mj_forward(model, data)
mujoco.mj_inverse(model, data)

udpated = False

plt.ion()
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(221, title="Joint Torques")




tau_lines = []
for i in range(22):
    tau_lines.append(ax.plot([], [], label=f'Joint {i}')[0])


imu_ax = fig.add_subplot(222, title="IMU: Possition Z")

rpy_ax = imu_ax.twinx()
imu_line = imu_ax.plot([], [],'y--', label='IMU')[0]
imu_data = []
rpy_data = []
rpy_line = []
for i in range(3):
    rpy_line.append(rpy_ax.plot([], [], label=f'RPY{i}')[0])

imu_ax.grid()
imu_ax.legend()
rpy_ax.legend()

ax.legend()
ax.grid()

qerr_ax = fig.add_subplot(223, title="Joint Constraint")
qerr_lines = []
qerr_data = []
ax_bar = fig.add_subplot(224, title="Joint Torques")

for i in range(22):
    qerr_lines.append(qerr_ax.plot([], [], label=f'Joint {i}')[0])
qerr_ax.legend()
qerr_ax.grid()

qrfc_act_data = []
qfrc_cons_data = []
x_data = []
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        # 设置目标位置
        
        step += 1
        mujoco.mj_step(model, data)
        viewer.sync()
        data.ctrl[:] = target_angles
        joint_acc = data.qacc.copy()
        max_acc = np.max(np.abs(joint_acc))

        if max_acc < 1e-5 and not udpated:
            #qpos = data.qpos.copy()
            #data.ctrl[:] = qpos[7:]
            #mujoco.mj_step(model, data)
            #viewer.sync()
            udpated = True
            #continue


        qfrc_inverse = data.qfrc_inverse.copy()[6:]
        
        qfrc_act = data.qfrc_actuator.copy()[6:]
        qfrc_bias = data.qfrc_bias.copy()[6:]
        qfrc_applied = data.qfrc_applied.copy()[6:]
        qfrc_constraint = data.qfrc_constraint.copy()[6:]
        qfrc_inverse = data.qfrc_inverse.copy()[6:]
        
        #print(data.qerr)
        #balance = qfrc_act - qfrc_bias + qfrc_applied
        #error = np.linalg.norm(qfrc_inverse - balance)
        #if step == 1:
        #    time.sleep(10)
        
        if step % 10 == 0:
            #print(f'step: {step}')
            #print("IMU:", end="\t")
            #print(*(f'{imu:.3f}' for imu in data.qpos[:7]), sep='\t')
            #print("Pos Error:", end='\t')
            #print(*(f'{err:.3f}' for err in list(target_angles - data.qpos[7:])), sep='\t') 
            #print("Joint Tau:", end='\t')
            #print(*(f'{err:.3f}' for err in data.qfrc_actuator[6:]), sep='\t')
            qerr_data.append(np.abs(target_angles - data.qpos[7:]))
            qrfc_act_data.append(np.abs(qfrc_act))
            qfrc_cons_data.append(np.abs(qfrc_constraint))
            x_data.append(step)
            

           
            imu_data.append(np.array(data.qpos[:6]))
            imu_line.set_data(x_data, np.array(imu_data)[:,2])
            imu_ax.relim()
            imu_ax.autoscale_view()
            for i, line in enumerate(tau_lines):
                line.set_data(x_data, np.array(qrfc_act_data)[:, i])
            for i, line in enumerate(qerr_lines):
                line.set_data(x_data, np.array(qfrc_cons_data)[:, i])

            qerr_ax.relim()
            qerr_ax.autoscale_view()
            ax.relim()
            ax.autoscale_view()


            Q = data.qpos[3:7]
            rpy = quaternion_to_euler_scipy(Q)
            rpy_data.append(rpy)

            for i, line in enumerate(rpy_line):
                line.set_data(x_data, np.array(rpy_data)[:, i])

            rpy_ax.relim()
            rpy_ax.autoscale_view()
            plt.pause(0.0001)
        
        if step == 300:
            categories = ['LF_J0', 'LF_J1', 'LF_J2', 'LF_J3', 'LF_J4', 'LM_J0', 'LM_J1', 'LM_J2', 'LR_J0', 'LR_J1', 'LR_J2','RR_J0', 'RR_J1', 'RR_J2', 'RM_J0', 'RM_J1', 'RM_J2', 'RF_J0', 'RF_J1', 'RF_J2', 'RF_J3', 'RF_J4']
            x = np.arange(len(categories))  # 类别位置
            width = 0.35  # 柱状图宽度
            ax_bar.bar(x - width/2,np.abs(qrfc_act_data[-1]), label='QFRC-ACT')
            print(f'max: {np.max(np.abs(qfrc_cons_data[-1]))}')
            ax_bar.bar(x + width/2,np.abs(qfrc_cons_data[-1]), label='QFRC-CONS')
            #print("Joint Tau:", end='\t')
            print(*(f'{err:s}' for err in categories), sep='\t')
            print(*(f'{err:.3f}' for err in qfrc_cons_data[-1]), sep='\t')
            ax_bar.grid()
            ax_bar.legend()

            ax_bar.relim()
            ax_bar.autoscale_view()
        #print(error)
        if step > 100000:
            break
            
        #data.qvel[:] = np.zeros(model.nv)
        #data.qpos[:] = qpos
        #data.qacc[:] = np.zeros(model.nv)


        #mujoco.mj_forward(model, data)
        
        #torques_inverse = data.qfrc_inverse.copy()

        #print(torques_inverse)
        #break
        #torques = get_all_joint_torques(model, data)
        
        #print(torques.shape)
        #print(torques)
        
        #viewer.sync()

plt.ioff()
plt.show()