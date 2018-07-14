import numpy as np
from KS_lib.image import KSimage


########################################################################################
def estimate_stain_matrix_macenko(I, I0=255, alpha=1.0, beta=0.15):
    I = I.astype(np.float32)
    I = I.reshape(-1, 3)

    # calculate optical density
    OD = -np.log((I + 1) / np.float(I0))

    # remove transparent pixels
    ODhat = OD[np.logical_not(np.any(OD < beta, 1)), :]

    # calculate eigenvectors
    E, V = np.linalg.eig(np.cov(ODhat.transpose()))

    ind = np.argsort(E)
    V = V[:, ind]  # second axis !!

    # project on the plane spanned by the eigenvectors corresponding to
    # the two largest eigenvalues
    theta = np.matmul(ODhat, V[:, 1:])
    phi = np.arctan2(theta[:, 1], theta[:, 0])

    # find the robust extremes(min and max angles)
    min_phi = np.percentile(phi, int(alpha))
    max_phi = np.percentile(phi, 100 - int(alpha))

    # Bring the extreme angles back to OD Space
    temp = np.vstack((np.cos(min_phi), np.sin(min_phi)))
    vec1 = np.matmul(V[:, 1:], temp)
    temp = np.vstack((np.cos(max_phi), np.sin(max_phi)))
    vec2 = np.matmul(V[:, 1:], temp)

    # Make sure that Hematoxylin is first and Eosin is second vector
    if vec1[0] > vec2[0]:
        M = np.hstack((vec1, vec2)).transpose()
    else:
        M = np.hstack((vec2, vec1)).transpose()

    return M


########################################################################################
def deconvolve(I, M=None, I0=255):
    I = I.astype(np.float32)
    assert I.ndim == 3, "Need an RGB image!"
    h, w, c = I.shape[0], I.shape[1], I.shape[2]
    assert (c == 3), "An image must be RGB!"

    if M is None:
        M = np.array([[0.644211, 0.716556, 0.266844], [0.092789, 0.954111, 0.283111]])

    # Add third Stain vector, if only two stain vectors are provided. This stain vector is obtained as the
    # cross product of first two stain vectors
    if M.shape[0] < 3:
        M = np.vstack((M, np.cross(M[0, :], M[1, :])))

    # Normalise the input so that each stain vector has a Euclidean norm of 1
    M = M / np.transpose(np.tile(np.sqrt(np.sum(M ** 2, axis=1)), (3, 1)))

    J = np.reshape(I, (-1, 3))

    OD = -np.log((J + 1.0) / np.float(I0))

    # determine concentrations of the individual stains
    C = np.transpose(np.linalg.solve(np.transpose(M), np.transpose(OD)))

    # stack back deconvolved channels
    deconvolved_ch = C.reshape(h, w, 3)

    return deconvolved_ch, M


########################################################################################
def stain_normalisation_macenko(source, target, alpha=1, beta=0.15):
    h, w = source.shape[0], source.shape[1]

    # Estimate Stain Matrix for Target Image
    if target.dtype == 'uint16':
        I0_target = 65535
    else:
        I0_target = 255
    M_target = estimate_stain_matrix_macenko(I=target, I0=I0_target, alpha=alpha, beta=beta)

    # Perform Color Deconvolution of Target Image to get stain concentration matrix
    C, M_target = deconvolve(target, M_target, I0_target)

    # Vectorize to Nx3matrix
    C = C.reshape(-1, 3)

    # Find the 99th percentile of stain concentration (for each channel)
    max_C_target = np.percentile(a=C, q=99, axis=0)

    # Repeat the same process for input / source image

    # Estimate Stain Matrix for Source Image
    if source.dtype == 'uint16':
        I0_source = 65535
    else:
        I0_source = 255
    M_source = estimate_stain_matrix_macenko(I=source, I0=I0_source, alpha=alpha, beta=beta)

    # Perform Color Deconvolution of Source Image to get stain concentration matrix
    C, M_source = deconvolve(source, M_source, I0_source)

    # Vectorize to Nx3 matrix
    C = C.reshape(-1, 3)

    # Find the 99th percentile of stain concentration(for each channel)
    max_C_source = np.percentile(a=C, q=99, axis=0)

    # main normalisation
    C = C / max_C_source
    C = C * max_C_target

    # Reconstruct the RGB image
    norm = I0_source * np.exp(np.matmul(C, -M_target)) - 1
    norm = norm.reshape(h, w, 3)
    norm = norm.clip(0, I0_source).astype(source.dtype)

    return norm

#####################################################################
target = KSimage.imread('1_421_1_2_7_999_11.jpg')
source = KSimage.imread('b001.tif')

norm = stain_normalisation_macenko(source, target)

KSimage.imwrite(norm, 'result.tiff')

print('done')
