import numpy as np

vec1 = np.array([6,7,3])
vec2 = np.array([6,7,3])
t = np.array([0,0,0.1])

def vector_similar(vec1,vec2,t):
    vec1_n = np.linalg.norm(vec1, ord=2)
    vec2_n = np.linalg.norm(vec2, ord=2)  # vector norm
    t_n = np.linalg.norm(t, ord=2)
    alpha = vec1_n / vec2_n
    theta = np.arccos(np.dot(vec1, vec2) / (vec1_n * vec2_n))
    omigax = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    omigay = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    omigaz = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    sigma = np.cos(theta)
    sigma_n = np.sin(theta)
    R = np.array([[sigma + (1 - sigma) * omigax * omigax, (1 - sigma) * omigax * omigay - omigaz * sigma_n,
                 (1 - sigma) * omigax * omigaz + omigay * sigma_n],
                 [(1 - sigma) * omigax * omigay + omigaz * sigma_n, sigma + (1 - sigma) * omigay * omigay,
                  (1 - sigma) * omigaz * omigay - omigax * sigma_n],
                 [(1 - sigma) * omigax * omigaz - omigay * sigma_n, (1 - sigma) * omigax * omigaz + omigay * sigma_n,
                  sigma + (1 - sigma) * omigaz * omigaz]])
    R_n = np.linalg.norm(R, ord=2)
    S = np.exp(-0.5 * np.abs(1 - alpha * R_n - t_n))
    #print(S)
    return S

#vector_similar(vec1,vec2,t)

