"""Conservative dark elongated-object check for the fixed wrist observation pose."""

import cv2
import numpy as np


def detect_vape(frame):
    h,w=frame.shape[:2]
    x0,x1=int(.05*w),int(.99*w)
    y0,y1=int(.04*h),int(.80*h)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    mask=cv2.inRange(gray,0,100)
    roi=np.zeros_like(mask);roi[y0:y1,x0:x1]=255
    mask=cv2.bitwise_and(mask,roi)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        if cv2.contourArea(contour)<500:continue
        (cx,cy),(a,b),_=cv2.minAreaRect(contour)
        if min(a,b)<8 or max(a,b)/min(a,b)<3:continue
        if not .15*w<max(a,b)<.8*w:continue
        x,y,bw,bh=cv2.boundingRect(contour)
        if x<=x0+2 or y<=y0+2 or x+bw>=x1-2 or y+bh>=y1-2:continue
        points=contour.reshape(-1,2).astype(float)
        _,vecs=np.linalg.eigh(np.cov(points.T))
        axis=vecs[:,-1]
        angle=float((np.degrees(np.arctan2(axis[1],axis[0]))+90)%180-90)
        candidates.append(dict(center=[float(cx),float(cy)],angle_deg=angle,
                               length_px=float(max(a,b)),width_px=float(min(a,b)),
                               box=cv2.boxPoints(cv2.minAreaRect(contour)).tolist(),
                               area=float(cv2.contourArea(contour))))
    candidates.sort(key=lambda c:c['area'],reverse=True)
    if not candidates:return None
    if len(candidates)>1 and candidates[1]['area']>.6*candidates[0]['area']:
        return None
    return candidates[0]


def compare_detection(reference,current):
    if reference is None or current is None:
        return dict(accepted=False,reason='Objet allongé unique non détecté.')
    distance=float(np.linalg.norm(np.array(reference['center'])-current['center']))
    angle=abs((current['angle_deg']-reference['angle_deg']+90)%180-90)
    scale=current['length_px']/reference['length_px']
    accepted=distance<=28 and angle<=6 and .85<=scale<=1.15
    return dict(accepted=bool(accepted),center_error_px=distance,angle_error_deg=angle,
                length_ratio=scale,reason='Alignement proche de la démonstration.' if accepted else 'Repositionner l’objet comme dans la vue de référence.')


def annotate(frame,detection):
    out=frame.copy()
    if detection:
        cv2.polylines(out,[np.int32(detection['box'])],True,(0,220,0),2)
        cv2.circle(out,tuple(np.int32(detection['center'])),4,(0,0,255),-1)
    return out
