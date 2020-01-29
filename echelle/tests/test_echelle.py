# from matplotlib.testing.decorators import image_comparison

# import echelle
# import numpy as np

# def _run_echelle(pandas=False, N=10000, seed=1234, ndim=3, factor=None,
#                 **kwargs):
#     np.random.seed(seed)
#     data1 = np.random.randn(ndim*4*N//5).reshape([4*N//5, ndim])
#     data2 = (5 * np.random.rand(ndim)[None, :] +
#              np.random.randn(ndim*N//5).reshape([N//5, ndim]))
#     data = np.vstack([data1, data2])
#     if factor is not None:
#         data[:, 0] *= factor
#         data[:, 1] /= factor
#     if pandas:
#         data = pd.DataFrame.from_items(zip(map("d{0}".format, range(ndim)),
#                                            data.T))

#     fig = corner.corner(data, **kwargs)
#     return fig


# # @image_comparison(baseline_images=["basic"], extensions=["png"])
# # def test_basic():
# #     _run_echelle()
