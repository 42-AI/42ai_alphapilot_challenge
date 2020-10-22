import numpy as np
import rospy
import math

class Gate(object):
    """docstring for Gate."""
    ## Constructor for class gate
    # @param: location: 4x3 array of corner locations
    # @param: inflation: Artificial inflation of gate to test for fly through
    def __init__(self, name, location, inflation):
        self.location = np.asarray(location)
        self.plane_equation = self.getPlaneOfGate()
        self.name = name
        minLoc = np.amin (self.location, axis=0)
        maxLoc = np.amax (self.location, axis=0)
        self.xmin = minLoc[0] - inflation
        self.xmax = maxLoc[0] + inflation
        self.ymin = minLoc[1] - inflation
        self.ymax = maxLoc[1] + inflation
        self.zmin = minLoc[2] - inflation
        self.zmax = maxLoc[2] + inflation

    # TODO: Engeristrer la valeur a l'initialisation, temps de calcule
    def getCenter(self):
        d = [
        sum([pos[0] for pos in self.location])/len(self.location),
        sum([pos[1] for pos in self.location])/len(self.location),
        sum([pos[2] for pos in self.location])/len(self.location)
        ]
        return d

    ## @brief Function to get the plane of the gate bounding box
    # @param self The object pointer
    def getPlaneOfGate(self):
        p1 = self.location[0,:]
        p2 = self.location[1,:]
        p3 = self.location[2,:]

        v1 = p3 - p1
        v2 = p2 - p1

        cp = np.cross(v1,v2)
        a,b,c = cp
        d = np.dot(cp, p3)
        return np.asarray([a,b,c,-d])

    ## @brief Function to get the distance of a point from plane
    # @param self The object pointer
    # @param point The query point to calcluate distance from
    def getDistanceFromPlane(self,point):
        d = math.fabs((self.plane_equation[0] * point[0] + self.plane_equation[1] * point[1] + self.plane_equation[2] * point[2] + self.plane_equation[3]))
        e = math.sqrt(self.plane_equation[0]**2 + self.plane_equation[1]**2 + self.plane_equation[2]**2)
        return (d/e)

    ## @brief Function to check if the drone is flying through a gate
    # @param self The object pointer
    # @param point The translation of the drone
    # @param tol The point to plane distance that is considered acceptable

    def isEvent(self, point1, point2):
        if not point1 or not point2:
            return False, False
        epsilon=1e-6
        point1 = [point1['x'], point1['y'], point1['z']]
        point2 = [point2['x'], point2['y'], point2['z']]
        rayPoint = point1
        rayDirection = np.array([point2[0]-point1[0],point2[1]-point1[1], point2[2]-point1[2]])
        planeNormal = np.array(self.getPlaneOfGate())
        planePoint = np.array(self.location[0])
    	ndotu = planeNormal[:3].dot(rayDirection)
    	if abs(ndotu) < epsilon:
    		return False, False
    	w = rayPoint - planePoint
    	si = -planeNormal[:3].dot(w) / ndotu
    	Psi = w + si * rayDirection + planePoint
        point = Psi.tolist()
        a = point1[0]*planeNormal[0]+point1[1]*planeNormal[1]+point1[2]*planeNormal[2]+planeNormal[3]
        b = point2[0]*planeNormal[0]+point2[1]*planeNormal[1]+point2[2]*planeNormal[2]+planeNormal[3]
        if (b > 0 if a < 0 else b < 0):
            if (point[0] < self.xmax) and (point[0] > self.xmin):
                if (point[1] < self.ymax) and (point[1] > self.ymin):
                    if (point[2] < self.zmax) and (point[2] > self.zmin):
                        return True, False
            if (point[0] < self.xmax+2) and (point[0] > self.xmin-2):
                if (point[1] < self.ymax+2) and (point[1] > self.ymin-2):
                    if (point[2] < self.zmax+2) and (point[2] > self.zmin-2):
                        return False, True
        return False, False
