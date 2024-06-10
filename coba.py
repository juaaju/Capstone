import matplotlib.pyplot as plt

sb_x = [1,2,3,1,1,2,4,5]
sb_y = [1,2,3,3,3,3,4,5]
pose = [1,3]

def tarik_garis(sb_y, pose_coords_y, sb_x):
    idx_y = []
    for idx, y in enumerate(sb_y):
        if pose_coords_y == y:
            idx_y.append(idx)
    
    x_val = []
    for i in idx_y:
        x_val.append(sb_x[i])

    lebar = abs(max(x_val) - min(x_val))
    print(x_val)
    return lebar
    
print(tarik_garis(sb_y, pose[1], sb_x))