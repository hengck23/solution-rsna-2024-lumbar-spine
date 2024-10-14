import cv2
import pandas as pd
import numpy as np

#custom augmentation
def do_resize_and_center(
    image, point, reference_size
):
    point = np.array(point, np.float32)
    H, W = image.shape[:2]
    if (W==reference_size) & (H==reference_size):
        return image, point

    s = reference_size / max(H, W)
    m = cv2.resize(image, dsize=None, fx=s, fy=s)
    h, w = m.shape[:2]
    padx0 = (reference_size-w)//2
    padx1 = reference_size-w-padx0
    pady0 = (reference_size-h)//2
    pady1 = reference_size-h-pady0

    m = np.pad(m, [[pady0, pady1], [padx0, padx1], [0, 0]], mode='constant', constant_values=0)
    p = point * s +[[padx0,pady0]]
    return m,p

def get_rotate_scale_by_reference_mat(
    point,
    image_shape,
    reference,
    scale_limit  = (-0.5,0.5),
    rotate_limit = (-45,45),
    shift_limit = (10,10),
    border=5
):
    H,W    = image_shape
    point  = np.array(point, dtype=np.float32)
    reference  = np.array(reference, dtype=np.float32)

    mat0, inlier0 = cv2.estimateAffinePartial2D(point, reference)
    point0 = np.concatenate([point, np.ones((len(point), 1))], axis=1) @ mat0.T
    mat1 = get_safe_rotate_scale_mat(
        point0,
        image_shape,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        shift_limit=shift_limit,
        border=border,
    )

    mat0 = np.concatenate([mat0,[[0,0,1]]])
    mat1 = np.concatenate([mat1,[[0,0,1]]])
    mat = mat1@mat0
    mat = mat[:2]

    return mat

def get_safe_custom_mat(
    point,
    image_shape,
    affline_limit = (-0.25,0.25),
    border=5
):
    H,W  = image_shape
    point  = np.array(point, dtype=np.float32)
    q = np.array([[x, y, 1] for x, y in point])

    src = np.array([
        [0,0],[0,H],[W,H],[W,0]
    ],dtype=np.float32)
    trial_state = 0
    trial = 0
    max_trial = 20
    while trial < max_trial:
        trial += 1
        size =max(H,W)
        dsrc = np.random.uniform(*affline_limit, (4,2))*size
        dst = src + dsrc
        dst = dst.astype(np.float32)
        #mat = cv2.getPerspectiveTransform(src, dst)
        mat, inliner = cv2.findHomography(src, dst)
        p = (q @ mat.T)
        p = p[:, :2] / p[:, [2]]

        xmin,xmax = p[:,0].min(), p[:,0].max()
        ymin,ymax = p[:,1].min(), p[:,1].max()
        if (xmin > border) & (xmax < W-border) & (ymin > border) & (ymax < H-border):
            trial_state=1
            break

    if trial_state==0:
        mat = np.array([
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ], dtype=np.float32).reshape(3, 3)
    return mat


#------------------------------------------------------
def get_safe_rotate_scale_mat(
    point,
    image_shape,
    scale_limit  = (-0.5,0.5),
    rotate_limit = (-45,45),
    shift_limit = (10,10),
    border=5
):
    H,W = image_shape
    point  = np.array(point, dtype=np.float32)
    mean   = point.mean(0,keepdims=True)
    mpoint = point-mean

    trial_state = 0
    trial=0
    max_trial=20
    while trial<max_trial:
        trial +=1
        scale = np.random.uniform(*scale_limit)+1
        rotate = np.random.uniform(*rotate_limit)
        cos = np.cos(rotate/180*np.pi)
        sin = np.sin(rotate/180*np.pi)
        mat = np.array([
            scale*cos, -scale*sin,
            scale*sin,  scale*cos,
        ]).reshape(2,2)

        p = mpoint@mat.T
        p = p-p.min(axis=0, keepdims=True)
        w,h = p.max(0)
        if (w > W-1.5-2*border) | (h > H-1.5-2*border):
            continue

        if shift_limit is None:
            shiftx = np.random.uniform(border,W-1.5-w-border)
            shifty = np.random.uniform(border,H-1.5-h-border)
        else:
            mx = (W-1.5-2*border-w)/2 + border
            my = (H-1.5-2*border-h)/2 + border
            shiftx = np.random.uniform(*scale_limit) + mx
            shifty = np.random.uniform(*scale_limit) + my

        p = p + [[shiftx, shifty]]
        p = p.astype(np.float32)
        mat, inliner = cv2.estimateAffinePartial2D(point, p)
        trial_state=1
        break

    if trial_state==0:
        mat = np.array([
            1, 0, 0,
            0, 1, 0,
        ], dtype=np.float32).reshape(2, 3)
    return mat

def apply_affine(
    image,point,mat
):
    H,W,D  = image.shape
    image_augment = cv2.warpAffine(image,mat,(W,H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    point = np.array([[x,y,1] for x,y in point])
    point_augment = (point @ mat.T).tolist()
    return image_augment, point_augment

def apply_perspective(
    image,point,mat
):
    H,W,D  = image.shape
    image_augment = cv2.warpPerspective(image,mat,(W,H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    point = np.array([[x,y,1] for x,y in point])
    point_augment = (point @ mat.T)
    point_augment = point_augment[:,:2]/ point_augment[:,[2]]
    point_augment = point_augment.tolist()
    return image_augment, point_augment
#########################################################################################################

def do_random_cutout(image,point):

    H,W,D  = image.shape
    point  = np.array(point, dtype=np.float32)
    xmin,xmax = point[:,0].min(), point[:,0].max()
    ymin,ymax = point[:,1].min(), point[:,1].max()

    w = np.random.randint(10,W)
    h = np.random.randint(1,ymin)
    x = np.random.randint(0,W-w)
    y = np.random.randint(0,ymin-h)
    image[y:y+h,x:x+w]=0

    return image
