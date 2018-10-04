from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np


def motion_update(particles, odom):
    """ Particle filter motion update
        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx, dy, dh) in *robot local frame*
        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    
    
    motion_particles = []
    dx,dy,dh = odom

    for particle in particles:
        odom_gayss =  add_odometry_noise(odom,ODOM_HEAD_SIGMA,ODOM_TRANS_SIGMA)
        dx,dy,dh = odom_gayss
        xr,yr = rotate_point(dx,dy,particle .h)
        x = particle .x + xr
        y = particle .y + yr
        h = particle .h + dh
        motion_particles.append(Particle(x,y,h))
    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update
        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once, and may not see any one
        grid -- grid world map, which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles
        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    p = np.array([1.0]*len(particles))

    for i,particle in enumerate(particles):
        if not grid.is_in(particle.x,particle.y): 
            p[i] = 0
            pass
        pmarkers = particle.read_markers(grid)
        lp = max(0, len(pmarkers)-len(measured_marker_list))
        lr = max(0, len(measured_marker_list)-len(pmarkers))

        if len(measured_marker_list) ==0 and len(pmarkers)==0:
            p[i] = 1
        elif len(measured_marker_list)==0 or len(pmarkers)==0:
            p[i] = DETECTION_FAILURE_RATE*SPURIOUS_DETECTION_RATE
        else:
            for rm in measured_marker_list:


                minDistance = None
                diffAngle = None

                for pm in pmarkers:

                    pm0, pm1, pm2 = pm[0],pm[1],pm[2]
                    d = grid_distance(rm[0],rm[1],pm0,pm1)

                    if (not minDistance) or (minDistance > d):
                        minDistance = d
                        diffAngle = diff_heading_deg(rm[2],pm2)

                power = - (minDistance**2)/(2*(MARKER_TRANS_SIGMA**2)) - (diffAngle**2)/(2*(MARKER_ROT_SIGMA**2))
                p[i] *= np.exp(power)

        p[i] = max(p[i], DETECTION_FAILURE_RATE*SPURIOUS_DETECTION_RATE)
        p[i] *= (DETECTION_FAILURE_RATE**lp)
        p[i] *= (SPURIOUS_DETECTION_RATE**lr)

    p /= sum(p)


    # Todo: resamplef
    rPercent = 0.015
    numRand = int(np.rint(rPercent*len(particles)))
    indexes = np.random.choice(a=range(0,len(particles)),size=(len(particles) - numRand),replace=True,p=p).tolist()
    measured_particles[0:numRand] = Particle.create_random(numRand,grid)
    measured_particles[numRand+1:-1] = [particles[i] for i in indexes]

    return measured_particles

