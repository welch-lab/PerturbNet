import numpy as np





def solve_y_axb(Y, A=None, B=None, A_ridge=0.1, B_ridge=0.1):
    if not isinstance(Y, np.ndarray):
        raise ValueError("Y must be a numpy array or matrix.")

    if A is not None and not isinstance(A, np.ndarray):
        raise ValueError("A must be None or a numpy array.")
    if B is not None and not isinstance(B, np.ndarray):
        raise ValueError("B must be None or a numpy array.")
    
    center = np.mean(Y, axis=1, keepdims=True)
    Y = Y - center

    if A is not None and B is not None:
        if Y.shape[0] != A.shape[0]:
            raise ValueError("Number of rows of Y must be equal to number of rows of A.")
        if Y.shape[1] != B.shape[1]:
            raise ValueError("Number of columns of Y must be equal to number of columns of B.")

        tmp = np.linalg.inv(A.T @ A + np.eye(A.shape[1]) * A_ridge) @ A.T @ Y @ B.T @ np.linalg.inv(B @ B.T + np.eye(B.shape[0]) * B_ridge)
    
    elif B is None:
        tmp = np.linalg.inv(A.T @ A + np.eye(A.shape[1]) * A_ridge) @ A.T @ Y

    elif A is None:
        tmp = Y @ B.T @ np.linalg.inv(B @ B.T + np.eye(B.shape[0]) * B_ridge)
    
    else:
        raise ValueError("Either A or B must be non-null")
    tmp = np.nan_to_num(tmp)

    return {"K": tmp, "center": center}