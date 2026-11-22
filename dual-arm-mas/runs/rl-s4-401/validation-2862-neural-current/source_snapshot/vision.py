"""RGB inspection camera and calibrated primitive-template pose adaptation.

The old mesh RGB checkpoints fail on the new colored primitive held geometry.
This adapter retains image segmentation -> calibrated shape-pose estimation,
using either explicit RGB color regions (debug/control condition) or a small
dual-arm fine-tune of the existing SFN RGB segmenter, followed by the actual
square-prism/ring templates.
No body IDs, depth, true part poses or true relative error enter estimation.
The camera mount is calibrated to B's encoder/FK grasp frame, not tracking A.
"""
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
from .world import wrap


class RGBPoseSensor:
    def __init__(self, size=768, segmentation_path=None, refine_boundaries=True):
        import pybullet as p
        self.size=size
        self.offset=np.array([0.,-.08,.04])
        self.view=np.asarray(p.computeViewMatrix(self.offset,[0,0,0],[0,0,1])).reshape(4,4,order="F")
        self.proj=np.asarray(p.computeProjectionMatrixFOV(40,1,.001,1)).reshape(4,4,order="F")
        self.matrix=self.proj@self.view
        self.previous=None
        self.segmentation=None
        self.segmentation_path=segmentation_path
        self.refine_boundaries=refine_boundaries
        if segmentation_path is not None:
            import torch
            from sfn.models.vsn import VirtualSensorNetwork
            self.device='cuda' if torch.cuda.is_available() else 'cpu'
            self.segmentation=VirtualSensorNetwork.from_checkpoints(segmentation_path=segmentation_path).segmentation.to(self.device).eval()

    def reset(self):self.previous=None

    def render(self,world):
        p=world.p
        # Physical camera mounting comes only from B encoder FK and known grasp.
        center,quat=world.expected_part_pose("B")
        yaw=p.getEulerFromQuaternion(quat)[2]
        R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        eye=np.asarray(center)+R@self.offset
        view=p.computeViewMatrix(eye,center,[0,0,1])
        raw=p.getCameraImage(self.size,self.size,view,self.proj.T.flatten().tolist(),
            renderer=p.ER_TINY_RENDERER,physicsClientId=world.client_id)
        return np.asarray(raw[2],dtype=np.uint8).reshape(self.size,self.size,4)[:,:,:3].copy(),yaw

    def segment(self,rgb):
        neural=None
        if self.segmentation is not None:
            import torch
            with torch.no_grad():
                image=torch.as_tensor(np.asarray(rgb).copy(),device=self.device,dtype=torch.float32).permute(2,0,1).unsqueeze(0)/255
                neural=self.segmentation(image).argmax(1)[0].cpu().numpy().astype('uint8')
            if not self.refine_boundaries:return neural
        image=np.asarray(rgb,dtype=float)
        r,g,b=image[:,:,0],image[:,:,1],image[:,:,2]
        # Ratios tolerate renderer shading; grayscale robots/background excluded.
        blue=(b>r*1.5)&(b>g*1.3)&(b>35)
        yellow=(r>g*.95)&(g>b*1.8)&(r>b*2)&(g>40)
        mask=np.zeros(r.shape,dtype=np.uint8);mask[blue]=1;mask[yellow]=2
        if neural is not None:
            # The learned semantic proposal localizes each component, then
            # calibrated RGB boundaries restore the sub-degree contour precision
            # lost to CNN pixel smoothing. There is NO global color fallback:
            # missing neural proposals remain missing, including on blank input.
            for k in (1,2):
                candidate=cv2.dilate((neural==k).astype('uint8'),np.ones((9,9),dtype='uint8'))>0
                mask[(mask==k)&~candidate]=0
        return mask

    def project(self,points):
        points=np.asarray(points)
        clip=np.c_[points,np.ones(len(points))]@self.matrix.T
        xy=clip[:,:2]/clip[:,3,None]
        return (xy*np.array([1,-1])+1)*self.size/2

    def unproject_plane(self,pixels,z):
        # Inverse planar homography with explicit metric plane height.
        H=self.matrix[:,[0,1,3]].copy();H[:,2]+=self.matrix[:,2]*z
        pix=np.asarray(pixels);ndc=np.c_[pix[:,0]*2/self.size-1,1-pix[:,1]*2/self.size,np.ones(len(pix))]
        ray=ndc@np.linalg.inv(H[[0,1,3],:]).T
        return ray[:,:2]/ray[:,2,None]

    def polygon(self,pose):
        x,y,yaw=pose;c,s=np.cos(yaw),np.sin(yaw)
        square=np.array([[-.005,-.005],[.005,-.005],[.005,.005],[-.005,.005]])@np.array([[c,s],[-s,c]])
        square+=np.array([x,y])
        vertices=np.vstack([np.c_[square,np.full(4,z)] for z in [.02,.08]])
        projected=self.project(vertices).astype(np.float32)
        return cv2.convexHull(projected).reshape(-1,2).astype(float)

    @staticmethod
    def edge_distances(points,polygon):
        start=polygon;end=np.roll(polygon,-1,axis=0);delta=end-start
        v=points[:,None,:]-start[None,:,:]
        t=np.clip(np.sum(v*delta[None,:,:],axis=2)/np.sum(delta*delta,axis=1),0,1)
        d=v-t[:,:,None]*delta[None,:,:]
        return np.sqrt(np.min(np.sum(d*d,axis=2),axis=1)+1e-8)

    def estimate(self,rgb,mount_yaw=0.):
        began=time.perf_counter();mask=self.segment(rgb)
        contours,_=cv2.findContours((mask==1).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        yellow=np.argwhere(mask==2)[:,::-1].astype(float)
        if not contours or len(yellow)<15:
            return {"valid":False,"reason":"missing_peg_or_ring","mask":mask,"error":None}
        contour=max(contours,key=cv2.contourArea).reshape(-1,2).astype(float)
        # Ignore upper image clipping and gripper-side occlusion. Bottom and
        # lateral silhouette constrain the known prism's translation and yaw.
        contour=contour[contour[:,1]>12]
        contour=contour[::max(1,len(contour)//160)]
        if len(contour)<10:return {"valid":False,"reason":"insufficient_outline","mask":mask,"error":None}
        rect=cv2.minAreaRect(self.unproject_plane(yellow,.0072).astype('float32'))
        hole=np.array(rect[0]);box=cv2.boxPoints(rect)
        edges=np.roll(box,-1,axis=0)-box
        ring_yaw=np.arctan2(edges[0,1],edges[0,0]);ring_yaw=(ring_yaw+np.pi/4)%(np.pi/2)-np.pi/4
        guesses=[self.previous] if self.previous is not None else [np.array([0,0,a]) for a in [-.18,0,.18]]
        def residual(pose):return self.edge_distances(contour,self.polygon(pose))
        def jacobian(pose):
            columns=[]
            for axis,step in enumerate([1e-5,1e-5,.002]):
                d=np.zeros(3);d[axis]=step
                columns.append((residual(pose+d)-residual(pose-d))/(2*step))
            return np.array(columns).T
        best=None
        for guess in guesses:
            fit=least_squares(residual,guess,jac=jacobian,
                bounds=([-.025,-.025,-.6],[.025,.025,.6]),x_scale=[.01,.01,.2],
                max_nfev=65,ftol=1e-5,xtol=1e-6,gtol=1e-5)
            score=float(np.mean(fit.fun**2))
            if best is None or score<best[0]:best=(score,fit.x)
        score,pose=best
        # Prism silhouette topology changes when a second lower edge becomes
        # visible. A single warm-start least-squares fit can stick on the wrong
        # branch; coarse yaw restarts only on a poor pixel fit resolve it.
        if score>.25:
            for yaw in np.arange(-.3,.3001,.05):
                guess=np.r_[pose[:2],yaw]
                fit=least_squares(residual,guess,jac=jacobian,
                    bounds=([-.025,-.025,-.6],[.025,.025,.6]),x_scale=[.01,.01,.2],
                    max_nfev=45,ftol=1e-5,xtol=1e-6,gtol=1e-5)
                candidate=float(np.mean(fit.fun**2))
                if candidate<score:score,pose=candidate,fit.x
        self.previous=pose.copy()
        xy=pose[:2]-hole;c,s=np.cos(mount_yaw),np.sin(mount_yaw)
        xy=np.array([[c,-s],[s,c]])@xy
        error=np.r_[xy,wrap(pose[2]-ring_yaw)]
        valid=bool(score<9 and np.all(np.abs(pose[:2])<.024))
        return {"valid":valid,"reason":None if valid else "poor_template_fit","mask":mask,"error":error,
            "template_residual_px2":score,"peg_pose_camera_frame":pose.tolist(),"hole_pose_camera_frame":[*hole.tolist(),float(ring_yaw)],
            "latency_seconds":time.perf_counter()-began,
            "source":("rendered_RGB_adapted_SFN_segmenter_"+("RGB_boundary_refinement_" if self.refine_boundaries else "")+"calibrated_prism_template") if self.segmentation is not None else "rendered_RGB_color_regions_calibrated_prism_template"}

    def observe(self,world):
        rgb,yaw=self.render(world)
        return rgb,self.estimate(rgb,yaw)
