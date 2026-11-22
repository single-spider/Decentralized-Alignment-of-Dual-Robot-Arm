"""Separate contact-enabled extension. Frozen alignment world is not modified.

Reset/IK may use joint resets while the clock is stopped. Execution exclusively
uses joint motors and one common stepSimulation; held parts use fixed constraints,
not per-tick pose resets. Fixed grasps are assumed fixtures, not frictional grasps.
"""
from dataclasses import asdict, dataclass
import numpy as np
from .world import DualArmWorld, WorldConfig


@dataclass
class ContactConfig:
    command_steps: int = 96
    force_limit_n: float = 8.
    penetration_limit_m: float = .0005
    grasp_drift_limit_m: float = .001
    target_depth_m: float = .020
    insertion_increment_m: float = .0005
    max_attempts: int = 100
    motor_forces: tuple = (87.,87.,87.,87.,12.,12.,12.)


class ContactWorld(DualArmWorld):
    def __init__(self, config=None):
        self.contact_config=config or ContactConfig()
        super().__init__(WorldConfig(gravity=(0.,0.,-9.81)))
        self.constraints={}
        self.motor_targets={n:self.q(n) for n in self.robots}
        self.p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=self.client_id)

    def attach_parts(self, blocked=False):
        if self.parts:raise RuntimeError('Parts already attached')
        p=self.p;cid=self.client_id
        self.grasps={'A':([0,0,.03],p.getQuaternionFromEuler([np.pi,0,0])),
                     'B':([0,-.28,0],p.getQuaternionFromEuler([np.pi,0,np.pi]))}
        boxes=[[.009,.025,.007],[.009,.025,.007],[.007,.009,.007],[.007,.009,.007],[.009,.128,.007]]
        positions=[[-.016,0,0],[.016,0,0],[0,-.016,0],[0,.016,0],[0,.153,0]]
        if blocked:
            boxes.append([.007,.007,.003]);positions.append([0,0,-.012])
        visual_a=p.createVisualShape(p.GEOM_BOX,halfExtents=[.005,.005,.03],rgbaColor=[.1,.3,.9,1],physicsClientId=cid)
        collision_a=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.005,.005,.03],physicsClientId=cid)
        visual_b=p.createVisualShapeArray([p.GEOM_BOX]*len(boxes),halfExtents=boxes,visualFramePositions=positions,
            rgbaColors=[[.2,.65,.4,1]]*len(boxes),physicsClientId=cid)
        collision_b=p.createCollisionShapeArray([p.GEOM_BOX]*len(boxes),halfExtents=boxes,collisionFramePositions=positions,physicsClientId=cid)
        seam=p.createVisualShapeArray([p.GEOM_BOX]*4,
            halfExtents=[[.0006,.0076,.0002]]*2+[[.0076,.0006,.0002]]*2,
            visualFramePositions=[[-.0076,0,.0072],[.0076,0,.0072],[0,-.0076,.0072],[0,.0076,.0072]],
            rgbaColors=[[1,.8,.1,1]]*4,physicsClientId=cid)
        for n,mass,visual,collision in [('A',.06,visual_a,collision_a),('B',.18,visual_b,collision_b)]:
            pos,quat=self.expected_part_pose(n)
            links={} if n=='A' else dict(linkMasses=[0.],linkCollisionShapeIndices=[-1],linkVisualShapeIndices=[seam],
                linkPositions=[[0,0,0]],linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],
                linkInertialFrameOrientations=[[0,0,0,1]],linkParentIndices=[0],linkJointTypes=[p.JOINT_FIXED],linkJointAxis=[[0,0,0]])
            body=p.createMultiBody(baseMass=mass,baseCollisionShapeIndex=collision,baseVisualShapeIndex=visual,
                basePosition=pos,baseOrientation=quat,physicsClientId=cid,**links)
            self.parts[n]=body
            p.changeDynamics(body,-1,lateralFriction=.4,restitution=0,collisionMargin=.0001,
                contactStiffness=20000,contactDamping=100,physicsClientId=cid)
            constraint=p.createConstraint(self.robots[n],self.ee_link,body,-1,p.JOINT_FIXED,[0,0,0],
                self.grasps[n][0],[0,0,0],parentFrameOrientation=self.grasps[n][1],
                childFrameOrientation=[0,0,0,1],physicsClientId=cid)
            p.changeConstraint(constraint,maxForce=2000,erp=.8,physicsClientId=cid)
            self.constraints[n]=constraint
            # The fixed grasp replaces hand/part collision; other contacts remain.
            for j in range(-1,p.getNumJoints(self.robots[n],physicsClientId=cid)):
                p.setCollisionFilterPair(body,self.robots[n],-1,j,0,physicsClientId=cid)
        self.blocked=blocked
        self.motor_targets={n:self.q(n) for n in self.robots}
        self._motors(self.motor_targets)

    def sync_parts(self):
        # Reset-only compatibility. Never invoked by advance/execute.
        for n,body in self.parts.items():
            self.p.resetBasePositionAndOrientation(body,*self.expected_part_pose(n),physicsClientId=self.client_id)
            self.p.resetBaseVelocity(body,[0,0,0],[0,0,0],physicsClientId=self.client_id)

    def attachment_drift(self):
        out={}
        for n in self.parts:
            expected=self.expected_part_pose(n);actual=self.part_pose(n)
            out[n]={'position_m':float(np.linalg.norm(actual[0]-expected[0])),
                    'angle_rad':float(2*np.arccos(np.clip(abs(np.dot(actual[1],expected[1])),0,1)))}
        return out

    def contact_state(self):
        contacts=self.p.getContactPoints(self.parts['A'],self.parts['B'],physicsClientId=self.client_id)
        return {'normal_force_n':float(sum(c[9] for c in contacts)),
                'axial_normal_force_n':float(sum(abs(c[7][2])*c[9] for c in contacts)),
                'penetration_m':max([max(0.,-c[8]) for c in contacts],default=0.),
                'contacts':[{'distance_m':c[8],'normal_force_n':c[9],'normal_on_B':list(c[7])} for c in contacts]}

    def insertion_state(self):
        p=self.p;cid=self.client_id;peg,quat=self.part_pose('A');hole,hquat=self.part_pose('B')
        tip,_=p.multiplyTransforms(peg,quat,[0,0,-.03],[0,0,0,1])
        local,_=p.multiplyTransforms(*p.invertTransform(hole,hquat),tip,[0,0,0,1])
        return {'tip_in_fixture_m':list(local),'depth_m':float(.007-local[2]),
                'error_m_m_rad':self.error().tolist(),'grasp_drift':self.attachment_drift(),**self.contact_state()}

    def _motors(self,targets):
        p=self.p;cid=self.client_id
        for n,q in targets.items():
            p.setJointMotorControlArray(self.robots[n],self.joints,p.POSITION_CONTROL,targetPositions=list(q),
                forces=self.contact_config.motor_forces,positionGains=[.15]*7,velocityGains=[1.]*7,physicsClientId=cid)
            p.setJointMotorControlArray(self.robots[n],[9,10],p.POSITION_CONTROL,targetPositions=[.01,.01],forces=[20,20],physicsClientId=cid)

    def advance(self,targets,steps=None):
        """No reset/sync calls: dynamic motors, measured contact every physics tick."""
        cfg=self.contact_config;self._motors(targets);self.motor_targets={n:np.array(q).copy() for n,q in targets.items()}
        peak=0.;penetration=0.;reason=None;sample=None
        for _ in range(steps or cfg.command_steps):
            self.p.stepSimulation(physicsClientId=self.client_id);self.tick+=1
            sample=self.insertion_state();peak=max(peak,sample['normal_force_n']);penetration=max(penetration,sample['penetration_m'])
            if sample['normal_force_n']>cfg.force_limit_n:reason='force_limit'
            elif sample['penetration_m']>cfg.penetration_limit_m:reason='penetration_limit'
            elif max(v['position_m'] for v in sample['grasp_drift'].values())>cfg.grasp_drift_limit_m:reason='grasp_drift'
            if reason:
                # Stop advancing the commanded descent. Hold the measured joints.
                self.motor_targets={n:self.q(n) for n in self.robots};self._motors(self.motor_targets);break
        return {'state':sample,'peak_normal_force_n':peak,'peak_penetration_m':penetration,'stop_reason':reason,'tick':self.tick}

    def execute(self,actions):
        before=self.measure()
        targets={n:self.prepare(n,np.asarray(actions.get(n,[0,0,0]),dtype=float)) for n in self.robots}
        result=self.advance(targets)
        if result['stop_reason']:raise ValueError('Contact safety stop: '+result['stop_reason'])
        return {'before':before,'after':self.measure(),'prepared_q':{n:q.tolist() for n,q in targets.items()},
                'tick':self.tick,'config':{'execution_mode':'dynamic_motors_fixed_grasps',**asdict(self.contact_config)},'contact':result}

    def descend(self):
        pos,quat=self.pose('A');target=pos.copy();target[2]-=self.contact_config.insertion_increment_m
        targets={'A':self.solve('A',target,quat),'B':self.motor_targets['B'].copy()}
        return self.advance(targets)


def insert(world, require_alignment=True, observer=None):
    """Bounded incremental axial descent with force/penetration/drift/stall guards.

    A performs axial descent; B independently holds its measured post-alignment
    target. This is a declared deterministic insertion stage, NOT trained axial RL.
    """
    cfg=world.contact_config;trace=[];initial=world.insertion_state();last=initial['depth_m'];stalls=0
    if require_alignment and (max(abs(x) for x in initial['error_m_m_rad'][:2])>.001 or abs(initial['error_m_m_rad'][2])>np.deg2rad(1)):
        return {'success':False,'reason':'alignment_gate','initial':initial,'trace':trace,'final':initial}
    world.motor_targets['B']=world.q('B').copy()
    for attempt in range(cfg.max_attempts):
        try:record=world.descend()
        except ValueError as exc:
            return {'success':False,'reason':'ik_failure','detail':str(exc),'initial':initial,'trace':trace,'final':world.insertion_state()}
        record['attempt']=attempt;trace.append(record);state=record['state']
        if observer is not None:observer(record)
        if record['stop_reason']:reason=record['stop_reason'];break
        # Geometry + low penetration/drift/force, not merely the requested Z.
        inside=max(abs(x) for x in state['tip_in_fixture_m'][:2])<=.001
        if state['depth_m']>=cfg.target_depth_m and inside and abs(state['error_m_m_rad'][2])<=np.deg2rad(1):
            return {'success':True,'reason':'inserted','initial':initial,'trace':trace,'final':state}
        progress=state['depth_m']-last;stalls=stalls+1 if progress<cfg.insertion_increment_m*.1 else 0;last=state['depth_m']
        if stalls>=3:reason='jam_stall';break
    else:reason='timeout'
    return {'success':False,'reason':reason,'initial':initial,'trace':trace,'final':world.insertion_state()}
