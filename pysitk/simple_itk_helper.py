##
# \file simple_itk_helper.py
# \brief      Utility functions associated to SimpleITK and ITK
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2015
#


import re
import os
import itk
import fnmatch
import datetime
import subprocess
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt

import pysitk.python_helper as ph
from pysitk.definitions import VIEWER
from pysitk.definitions import DIR_TMP
from pysitk.definitions import ITKSNAP_EXE, FSLVIEW_EXE, NIFTYVIEW_EXE

# Use ITK-SNAP instead of imageJ to view images
os.environ['SITK_SHOW_COMMAND'] = ITKSNAP_EXE


TRANSFORM_SITK_DOF_LABELS_LONG = {
    6: ["angle_x [rad]",
        "angle_y [rad]",
        "angle_z [rad]",
        "t_x [mm]",
        "t_y [mm]",
        "t_z [mm]"],
}
TRANSFORM_SITK_DOF_LABELS_SHORT = {
    6: ["angle_x",
        "angle_y",
        "angle_z",
        "t_x",
        "t_y",
        "t_z"],
}


##
# Get composite transform of two affine/euler sitk transforms
# \see        http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/22_Transforms.html
# \date       2017-08-14 11:51:32+0100
#
# \param      transform_outer  The transform outer
# \param      transform_inner  The transform inner
#
# \return     The composite sitk affine/euler transform.
#
def get_composite_sitk_affine_transform(transform_outer, transform_inner):

    dim = transform_outer.GetDimension()

    A_inner = np.asarray(transform_inner.GetMatrix()).reshape(dim, dim)
    c_inner = np.asarray(transform_inner.GetCenter())
    t_inner = np.asarray(transform_inner.GetTranslation())

    A_outer = np.asarray(transform_outer.GetMatrix()).reshape(dim, dim)
    c_outer = np.asarray(transform_outer.GetCenter())
    t_outer = np.asarray(transform_outer.GetTranslation())

    A_composite = A_outer.dot(A_inner)
    c_composite = c_inner
    t_composite = A_outer.dot(
        t_inner + c_inner - c_outer) + t_outer + c_outer - c_inner

    if transform_outer.GetName() == "AffineTransform" \
            or transform_inner.GetName() == "AffineTransform" \
            or transform_outer.GetName() != transform_inner.GetName():
        trafo = sitk.AffineTransform(dim)
    else:
        trafo = eval("sitk." + transform_outer.GetName() + "()")

    trafo.SetMatrix(A_composite.flatten())
    trafo.SetTranslation(t_composite)
    trafo.SetCenter(c_composite)

    return trafo


##
# Composite two Euler Transforms
# \param[in]  transform_outer  as sitk::simple::EulerxDTransform
# \param[in]  transform_inner  as sitk::simple::EulerxDTransform
# \return     \p tranform_outer
# \f$ \circ
# \f$ \p transform_inner as sitk.EulerxDTransform
# \see        http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/22_Transforms.html
#
def get_composite_sitk_euler_transform(transform_outer, transform_inner):

    # Guarantee type sitk::simple::AffineTransform of transformations
    # transform_outer = sitk.AffineTransform(transform_outer)
    # transform_inner = sitk.AffineTransform(transform_inner)

    dim = transform_outer.GetDimension()

    A_inner = np.asarray(transform_inner.GetMatrix()).reshape(dim, dim)
    c_inner = np.asarray(transform_inner.GetCenter())
    t_inner = np.asarray(transform_inner.GetTranslation())

    A_outer = np.asarray(transform_outer.GetMatrix()).reshape(dim, dim)
    c_outer = np.asarray(transform_outer.GetCenter())
    t_outer = np.asarray(transform_outer.GetTranslation())

    A_composite = A_outer.dot(A_inner)
    c_composite = c_inner
    t_composite = A_outer.dot(
        t_inner + c_inner - c_outer) + t_outer + c_outer - c_inner

    euler = eval("sitk." + transform_outer.GetName() + "()")
    euler.SetMatrix(A_composite.flatten())
    euler.SetTranslation(t_composite)
    euler.SetCenter(c_composite)

    return euler


##
# Get direction for sitk.Image object from sitk.AffineTransform instance. The
# information of the image is required to extract spacing information and
# associated image dimension
# \param[in]  affine_transform_sitk  sitk.AffineTransform instance
# \param[in]  image_or_spacing_sitk  provide entire image as sitk object or
#                                    spacing directly
# \return     image direction which can be used to update the sitk.Image via
#             image_sitk.SetDirection(direction)
#
def get_sitk_image_direction_from_sitk_affine_transform(affine_transform_sitk,
                                                        image_or_spacing_sitk):
    dim = affine_transform_sitk.GetDimension()
    try:
        spacing_sitk = np.array(image_or_spacing_sitk.GetSpacing())
    except:
        spacing_sitk = np.array(image_or_spacing_sitk)

    S_inv_sitk = np.diag(1 / spacing_sitk)

    A = np.array(affine_transform_sitk.GetMatrix()).reshape(dim, dim)

    return A.dot(S_inv_sitk).flatten()


##
# Get origin for sitk.Image object from sitk.AffineTransform instance. The
# information of the image is required to extract spacing information and
# associated image dimension
# \param[in]  affine_transform_sitk  sitk.AffineTransform instance
# \param[in]  image_sitk             image as sitk.Image object sought to be
#                                    updated
# \return     image origin which can be used to update the sitk.Image via
#             image_sitk.SetOrigin(origin) TODO: eliminate image_sitk from the
#             header
#
def get_sitk_image_origin_from_sitk_affine_transform(affine_transform_sitk,
                                                     image_sitk=None):
    """
    Important: Only tested for center=\0! Not clear how it shall be implemented,
            cf. Johnson2015a on page 551 vs page 107!

    Mostly outcome of application of get_composite_sitk_affine_transform and first transform_inner is image.
    Therefore, center_composite is always zero on tested functions so far
    """
    # dim = len(image_sitk.GetSize())
    dim = affine_transform_sitk.GetDimension()

    affine_center = np.array(affine_transform_sitk.GetCenter())
    affine_translation = np.array(affine_transform_sitk.GetTranslation())

    R = np.array(affine_transform_sitk.GetMatrix()).reshape(dim, dim)

    # return affine_center + affine_translation
    return affine_center + affine_translation - R.dot(affine_center)


##
# Gets the sitk affine transform from sitk image, to transform an index to the
# physical space
# \date       2016-11-06 18:54:09+0000
#
# Returns the affine transform of an sitk image which links the voxel space
# (i.e. its data array elements) with the p positions in the physical space.
# I.e. it transforms a voxel index to the physical coordinate:
# \f[ T(\vec{i}) = R\,S\,\vec(i) + \vec{o} = \vec{x}
# \f] with
# \f$R\f$ being the rotation matrix (direction matrix via \p GetDirection),
# \f$S=diag(s_i)\f$ the scaling matrix,
# \f$\vec{i}\f$ the voxel index,
# \f$\vec{o}\f$ the image origin (via \p GetOrigin) and
# \f$\vec{x}\f$ the final physical coordinate.
#
# \param      image_sitk  The image sitk
#
# \return     Transform T(i) = R*S*i + origin as sitk.AffineTransform
#
def get_sitk_affine_transform_from_sitk_image(image_sitk):

    # Matrix A = R*S
    A = get_sitk_affine_matrix_from_sitk_image(image_sitk)

    # Translation t = origin
    t = np.array(image_sitk.GetOrigin())

    # T(i) = R*S*i + origin
    return sitk.AffineTransform(A, t)


##
# Copy sitk-type transform and return same type
# \date       2018-04-18 22:51:51-0600
#
# \param      transform_sitk  Transform as sitk.Transform
#
# \return     Same-type copy of an sitk.Transform
#
def copy_transform_sitk(transform_sitk):

    name = transform_sitk.GetName()
    transform_sitk_copy = eval("sitk.%s" % name)(transform_sitk)

    return transform_sitk_copy


def read_transform_sitk(path_to_file, inverse=False):
    transform_types = {
        "Euler2DTransform_double_2_2": sitk.Euler2DTransform,
        "Euler3DTransform_double_3_3": sitk.Euler3DTransform,
        "AffineTransform_double_2_2": sitk.AffineTransform,
        "AffineTransform_double_3_3": sitk.AffineTransform,
        "MatrixOffsetTransformBase_double_3_3": sitk.AffineTransform,
    }

    # Used for to/from FLIRT transform conversion
    transform_types_dim = {
        "MatrixOffsetTransformBase_double_3_3": 3,
    }

    # read as sitk.Transform
    transform_sitk = sitk.ReadTransform(path_to_file)

    # convert to correct type of transform, e.g. Euler3DTransform
    transform_type = ph.read_file_line_by_line(path_to_file)[2]
    transform_type = re.sub("\n", "", transform_type)
    transform_type = transform_type.split(" ")[1]
    if transform_type in ["MatrixOffsetTransformBase_double_3_3"]:
        transform_sitk_ = sitk.AffineTransform(
            transform_types_dim[transform_type])
        transform_sitk_.SetParameters(transform_sitk.GetParameters())
        transform_sitk_.SetFixedParameters(transform_sitk.GetFixedParameters())
        transform_sitk = transform_sitk_
    else:
        transform_sitk = transform_types[transform_type](transform_sitk)

    # invert transform
    if inverse:
        transform_sitk = transform_types[transform_type](
            transform_sitk.GetInverse())

    return transform_sitk


def read_transform_itk(path_to_file, inverse=False, pixel_type=itk.D):
    transform_sitk = read_transform_sitk(path_to_file, inverse=inverse)
    return get_itk_from_sitk_transform(transform_sitk, pixel_type=pixel_type)


def invert_transform_sitk(transform_sitk):
    return getattr(sitk, transform_sitk.GetName())(transform_sitk.GetInverse())


##
# Gets the sitk affine matrix from sitk image.
# \date       2016-11-06 19:07:03+0000
#
# \param      image_sitk  The image sitk
#
# \return     Matrix R*S as flattened data array ready for SetMatrix
#
def get_sitk_affine_matrix_from_sitk_image(image_sitk):
    dim = len(image_sitk.GetSize())
    spacing_sitk = np.array(image_sitk.GetSpacing())
    S_sitk = np.diag(spacing_sitk)
    R_sitk = np.array(image_sitk.GetDirection()).reshape(dim, dim)

    return R_sitk.dot(S_sitk).flatten()


##
# Get sitk.AffineTransform object based on direction and origin. The idea is,
# to get the affine transform which describes the physical position of the
# respective image. The information of the image is required to extract spacing
# information and associated image dimension
# \param[in]  direction_sitk         direction obtained via GetDirection() of
#                                    sitk.Image or similar
# \param[in]  origin_sitk            origin obtained via GetOrigin() of
#                                    sitk.Image or similar
# \param[in]  image_or_spacing_sitk  provide entire image as sitk object or
#                                    spacing directly
# \return     Affine transform as sitk.AffineTransform object
#
def get_sitk_affine_transform_from_sitk_direction_and_origin(
        direction_sitk,
        origin_sitk,
        image_or_spacing_sitk):
    dim = len(origin_sitk)
    try:
        spacing_sitk = np.array(image_or_spacing_sitk.GetSpacing())
    except:
        spacing_sitk = np.array(image_or_spacing_sitk)

    S_sitk = np.diag(spacing_sitk)

    direction_matrix_sitk = np.array(direction_sitk).reshape(dim, dim)
    affine_matrix_sitk = direction_matrix_sitk.dot(S_sitk).flatten()

    return sitk.AffineTransform(affine_matrix_sitk, origin_sitk)


##
# rigid_transform_*D (object type  Transform) as output of object
# sitk.ImageRegistrationMethod does not contain the member functions GetCenter,
# GetTranslation, GetMatrix whereas the objects sitk.Euler*DTransform does.
# Hence, create an instance sitk.Euler*D so that it can be used for composition
# of transforms as coded in get_composite_sitk_affine_transform
# \date       2017-06-26 17:07:42+0100
#
# \param      rigid_registration_transform  The rigid registration transform
#
# \return     The inverse of sitk rigid registration transform.
#
def get_inverse_of_sitk_rigid_registration_transform(
        rigid_registration_transform):

    dim = rigid_registration_transform.GetDimension()

    if dim == 2:
        rigid_transform_2D = rigid_registration_transform

        # Steps could have been chosen the same way as in the 3D case. However,
        # here the computational steps more visible

        # Extract parameters of 2D registration
        angle, translation_x, translation_y = rigid_transform_2D.GetParameters()
        center = rigid_transform_2D.GetFixedParameters()

        # Create transformation used to align moving -> fixed

        # Obtain inverse translation
        tmp_trafo = sitk.Euler2DTransform((0, 0), -angle, (0, 0))
        translation_inv = tmp_trafo.TransformPoint(
            (-translation_x, -translation_y))

        # Create instance of Euler2DTransform based on inverse = R_inv(x-c) -
        # R_inv(t) + c
        return sitk.Euler2DTransform(center, -angle, translation_inv)

    elif dim == 3:
        rigid_transform_3D = rigid_registration_transform

        # Create inverse transform of type Transform
        rigid_transform_3D_inv = rigid_transform_3D.GetInverse()

        # Extract parameters of inverse 3D transform to feed them back to
        # object Euler3DTransform:
        angle_x, angle_y, angle_z, translation_x, translation_y, translation_z = rigid_transform_3D_inv.GetParameters()
        center = rigid_transform_3D_inv.GetFixedParameters()

        # Return inverse of rigid_transform_3D as instance of Euler3DTransform
        return sitk.Euler3DTransform(center, angle_x, angle_y, angle_z, (translation_x, translation_y, translation_z))


##
# Gets the altered field of view sitk image.
# \date       2017-04-08 22:20:40+0100
#
# \param      image_sitk  The image sitk
# \param      boundary_i  added value to first coordinate (can also be
#                         negative)
# \param      boundary_j  added value to second coordinate (can also be
#                         negative)
# \param      boundary_k  added value to third coordinate (can also be
#                         negative)
# \param      unit        Unit can either be "mm" or "voxel"
#
# \return     sitk.Image with altered field of view, i.e. with shape_new[i] =
#             shape[i] + 2*boundary_voxel[i]
#
def get_altered_field_of_view_sitk_image(image_sitk,
                                         boundary_i=0,
                                         boundary_j=0,
                                         boundary_k=0,
                                         unit="mm"):

    size = np.array(image_sitk.GetSize()).astype("int")
    origin = np.array(image_sitk.GetOrigin())
    spacing = np.array(image_sitk.GetSpacing())
    dimension = image_sitk.GetDimension()

    # Get unit vectors of respective image dimension
    if dimension is 2:
        e_x = np.array([1, 0])
        e_y = np.array([0, 1])
    elif dimension is 3:
        e_x = np.array([1, 0, 0])
        e_y = np.array([0, 1, 0])
        e_z = np.array([0, 0, 1])

    # Dimension is given in mm
    if unit == "mm":
        boundary_i_voxel = np.round(boundary_i / spacing[0])
        boundary_j_voxel = np.round(boundary_j / spacing[1])

        # compute new shape of image
        size[0] += 2 * boundary_i_voxel
        size[1] += 2 * boundary_j_voxel

        if dimension is 3:
            boundary_k_voxel = np.round(boundary_k / spacing[2])

            size[2] += 2 * boundary_k_voxel

    # Dimension is given in voxels
    elif unit == "voxel":
        # compute new shape of image
        size[0] += 2 * boundary_i
        size[1] += 2 * boundary_j

        # express change in mm
        boundary_i *= spacing[0]
        boundary_j *= spacing[1]

        if dimension is 3:
            size[2] += 2 * boundary_k
            boundary_k *= spacing[2]

    else:
        raise ValueError("Unit can either be 'mm' or 'voxel'.")

    # Compute new origin so that image intensity information is not altered
    # in the physical space
    a_x = image_sitk.TransformIndexToPhysicalPoint(e_x) - origin
    a_y = image_sitk.TransformIndexToPhysicalPoint(e_y) - origin

    translation = a_x / np.linalg.norm(a_x) * boundary_i
    translation += a_y / np.linalg.norm(a_y) * boundary_j

    if dimension is 3:
        a_z = image_sitk.TransformIndexToPhysicalPoint(e_z) - origin
        translation += a_z / np.linalg.norm(a_z) * boundary_k

    origin = origin - translation

    # Resample image to new space, i.e. just change shape without changing
    # the image in the physical space
    image_sitk = sitk.Resample(
        image_sitk,
        size,
        eval("sitk.Euler" + str(dimension) + "DTransform()"),
        sitk.sitkNearestNeighbor,
        origin,
        spacing,
        image_sitk.GetDirection()
    )

    return image_sitk


#
# Get transformed (deepcopied) image
# \date       2017-06-26 17:06:25+0100
#
# \param[in]  image_init_sitk  image as sitk.Image object to be transformed
# \param[in]  transform_sitk   transform to be applied as sitk.AffineTransform
#                              object
#
# \return     transformed image as sitk.Image object
#
def get_transformed_sitk_image(image_init_sitk, transform_sitk):
    image_sitk = sitk.Image(image_init_sitk)

    affine_transform_sitk = get_sitk_affine_transform_from_sitk_image(
        image_sitk)

    transform_sitk = get_composite_sitk_affine_transform(
        transform_sitk, affine_transform_sitk)
    # transform_sitk =
    # get_composite_sitk_affine_transform(get_inverse_of_sitk_rigid_registration_transform(affine_transform_sitk),
    # affine_transform_sitk)

    direction = get_sitk_image_direction_from_sitk_affine_transform(
        transform_sitk, image_sitk)
    origin = get_sitk_image_origin_from_sitk_affine_transform(
        transform_sitk, image_sitk)

    image_sitk.SetOrigin(origin)
    image_sitk.SetDirection(direction)

    return image_sitk


##
# Read image from file and return as ITK object
# \date       2017-06-26 17:05:40+0100
#
# \param[in]  filename    filename of image to read
# \param[in]  pixel_type  itk pixel types, like itk.D, itk.F, itk.UC etc
# \param[in]  dim         The dim
#
# \example    read_itk_image("image.nii.gz", itk.D, 3) to read image stack
# \example    read_itk_image("mask.nii.gz", itk.UC, 3) to read image stack mask
#
# \return     { description_of_the_return_value }
#
def read_itk_image(filename):
    return itk.imread(filename)


##
# Reads 3D vector image and return as SimpleITK image
# \date       2016-09-20 15:31:05+0100
#
# \param      filename             The filename
# \param      return_vector_index  Index/Component of vector image to be
#                                  returned. If 'None' the entire vector image
#                                  is returned
#
# \return     (Multi-component) sitk.Image object
#
def read_sitk_vector_image(filename, dtype=np.float64):

    # Workaround: Read vector image via nibabel
    image_nib = nib.load(filename)
    nda_nib = image_nib.get_data()
    nda_nib_shape = nda_nib.shape
    nda = np.zeros((nda_nib_shape[2],
                    nda_nib_shape[1],
                    nda_nib_shape[0],
                    nda_nib_shape[3]),
                   dtype=dtype)

    # Convert to (Simple)ITK data array format, i.e. reorder to
    # z-y-x-components shape
    for i in range(0, nda_nib_shape[2]):
        for k in range(0, nda_nib_shape[0]):
            nda[i, :, k, :] = nda_nib[k, :, i, :]

    # Get SimpleITK image
    vector_image_sitk = sitk.GetImageFromArray(nda)

    # Workaround: Update header from nibabel information
    R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    affine_nib = image_nib.affine
    R_nib = affine_nib[0:-1, 0:-1]

    # Get spacing (only for image dimensions, i.e. not for the vector
    # component)
    spacing_sitk = np.array(image_nib.header.get_zooms())[0:R_nib.shape[0]]
    S_nib_inv = np.diag(1 / spacing_sitk)

    direction_sitk = R.dot(R_nib).dot(S_nib_inv).flatten()

    t_nib = affine_nib[0:-1, 3]
    origin_sitk = R.dot(t_nib)

    vector_image_sitk.SetSpacing(np.array(spacing_sitk).astype('double'))
    vector_image_sitk.SetDirection(direction_sitk)
    vector_image_sitk.SetOrigin(origin_sitk)

    return vector_image_sitk

    # All the other stuff did not work (neither in ITK nor in SimpleITK)! See
    # below. Readings worked but the two components always contained the same
    # value!
    # IMAGE_TYPE = itk.Image.D3
    # # IMAGE_TYPE = itk.Image.VD23
    # # IMAGE_TYPE = itk.VectorImage.D3
    # # IMAGE_TYPE = itk.Image.VF23
    # reader = itk.ImageFileReader[IMAGE_TYPE].New()
    # reader.SetFileName(DIR_INPUT + filename_ref + ".nii")
    # reader.Update()
    # foo_itk = reader.GetOutput()


##
# Extract single component from vector image
# \date       2017-08-06 16:58:57+0100
#
# \param      vector_image_sitk  Vector image as sitk.Image
# \param      component          Index/Component of vector image to be
#                                  returned.
#
# \return     Component of vector image returned as sitk.Image
#
def extract_component_from_vector_image(vector_image_sitk, component):
    return sitk.VectorIndexSelectionCast(vector_image_sitk, component)


##
# Gets the sitk vector image from single components.
# \date       2017-08-06 17:01:17+0100
#
# \param      image_components_sitk  List of individual sitk.Image objects
#                                    which shall be combined to vector image
#
# \return     Multi-component sitk.Image object.
#
def get_sitk_vector_image_from_components(image_components_sitk):

    N_components = len(image_components_sitk)
    shape = image_components_sitk[0].GetSize()

    vector_image_nda = np.zeros((shape[2], shape[1], shape[0], N_components))
    for i in range(N_components):
        vector_image_nda[:, :, :, i] = sitk.GetArrayFromImage(
            image_components_sitk[i])

    vector_image_sitk = sitk.GetImageFromArray(vector_image_nda)

    vector_image_sitk.SetSpacing(image_components_sitk[0].GetSpacing())
    vector_image_sitk.SetDirection(image_components_sitk[0].GetDirection())
    vector_image_sitk.SetOrigin(image_components_sitk[0].GetOrigin())

    return vector_image_sitk


##
# Writes a sitk vector image.
# \date       2017-08-06 18:18:06+0100
#
# \param      vector_image_sitk  Vector image as sitk.Object
# \param      filename           filename path to write image ("nii" or
#                                ".nii.gz")
#
def write_sitk_vector_image(vector_image_sitk, filename):
    R = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]])
    origin_sitk = vector_image_sitk.GetOrigin()
    direction_sitk = vector_image_sitk.GetDirection()
    spacing_sitk = vector_image_sitk.GetSpacing()

    t_nib = R.dot(origin_sitk)
    R_nib = R.dot(np.array(direction_sitk).reshape(
        3, 3)).dot(np.diag(spacing_sitk))

    A_nib = np.eye(4)
    A_nib[0:-1, 3] = t_nib
    A_nib[0:-1, 0:-1] = R_nib

    nda = sitk.GetArrayFromImage(vector_image_sitk)
    shape = nda.shape
    nda_nib = np.zeros((shape[2], shape[1], shape[0], shape[3]))

    # Convert to Nibabel data array format, i.e. reorder to x-y-z-components
    # shape
    for i in range(0, nda.shape[2]):
        for k in range(0, nda.shape[0]):
            nda_nib[i, :, k, :] = nda[k, :, i, :]

    image_nib = nib.Nifti1Pair(nda_nib, A_nib)
    nib.save(image_nib, filename)
    ph.print_info("Vector image written to '%s'." % (filename))


##
# Writes a SimpleITK image object to NiftI file including both q- and s-forms.
#
# By default, ITK only writes the q-form and s-form is set to zero. The problem
# is that, e.g., ITK seems to prioritize the q-form whereas FSL prioritizes the
# s-form.
# \see        https://github.com/ANTsX/ANTs/wiki/How-does-ANTs-handle-qform-and-sform-in-NIFTI-1-images%3F
# \date       2017-11-03 16:03:10+0000
#
# \param      image_sitk    Image as sitk.Image object
# \param      path_to_file  path to filename
#
def write_nifti_image_sitk(image_sitk, path_to_file, verbose=0):

    ph.create_directory(os.path.dirname(path_to_file))
    if verbose:
        ph.print_info("Image written to '%s' ... " % path_to_file, newline=0)
    sitk.WriteImage(image_sitk, path_to_file)

    # Use fslorient to copy q-form to s-form. However, in case of a 3D slice,
    # it would set dim0 = 2 incorrectly. Using fslmodhd for such a case seems
    # to do the trick as it updates the s-form as well.
    if image_sitk.GetDimension() == 3 and image_sitk.GetSize()[-1] == 1:
        # Do not apply header update to single slice; causes troubles for
        # some FSL versions (e.g. 5.0.9)
        flag = 0
        # flag = ph.execute_command(
        #     "fslmodhd %s dim0 3" % path_to_file, verbose=debug)
    else:
        flag = apply_fslorient(path_to_file)

    if flag != 0:
        ph.print_warning(
            "Only q-form is set as fslorient was not successful!")

    if verbose:
        print("done")


def write_nifti_image_itk(image_itk, path_to_file, verbose=0):
    ph.create_directory(os.path.dirname(path_to_file))
    if verbose:
        ph.print_info("Image written to '%s' ... " % path_to_file, newline=0)
    itk.imwrite(image_itk, path_to_file)
    flag = apply_fslorient(path_to_file)

    if verbose:
        print("done")

    return flag


def write_nifti_image_nib(image_nib, path_to_file, verbose=0):

    ph.create_directory(os.path.dirname(path_to_file))
    if verbose:
        ph.print_info("Image written to '%s' ... " % path_to_file, newline=0)
    nib.save(image_nib, path_to_file)

    flag = apply_fslorient(path_to_file)

    if flag != 0:
        ph.print_warning(
            "Only q-form is set as fslorient was not successful!")

    if verbose:
        print("done")


##
# Update image header so that both s- and q-form information is set.
#
# By default, ITK only writes the q-form and s-form is set to zero. The problem
# is that, e.g., ITK seems to prioritize the q-form whereas FSL prioritizes the
# s-form.
# \see        https://github.com/ANTsX/ANTs/wiki/How-does-ANTs-handle-qform-and-sform-in-NIFTI-1-images%3F
#
# subprocess is used to wait for the process to be finished. Resulting image
# header format will/should be in LPI, i.e neurological (right-handed), rather
# than RPI, i.e. radiological (left-handed) orientation
# \date       2019-02-23 23:44:12+0000
#
# \param      path_to_file  The path to file
#
# \return     exit status
#
def apply_fslorient(path_to_file, verbose=False):
    # TODO: Depending on the NIfTI image, either s- or q-form is set but not
    # necessarily both. 'fslorient forceneurological/forceradiological' would
    # do the trick to set them regardless of whether s- or q-form is given but
    # may have unintended consequences since not properly tested.

    # flag = subprocess.call(["fslorient", "-forceradiological", path_to_file])
    # flag = subprocess.call(["fslorient", "-forceneurological", path_to_file])

    # Causes memory error for NiftyMIC virtual machine (avoid for now)
    # flag = subprocess.call(["fslorient", "-copyqform2sform", path_to_file])

    flag = ph.execute_command(
        "fslorient -copyqform2sform %s" % path_to_file, verbose=verbose)

    return flag


##
# Reads a nifti image and returns sitk.Image object.
#
# Potential nan and inf values are replaced by numerical values
# Remark: Not tested for vector images
# \date       2018-02-09 00:21:59+0000
#
# \param      file_path    The file path as string
# \param      pixel_type   The pixel type as sitk object
# \param      replace_nan  boolean to indicate whether nan should be replaced
#
# \return     Nifti image as sitk.Object
#
def read_nifti_image_sitk(
        file_path,
        pixel_type=sitk.sitkUnknown,
        replace_nan=1):
    image_sitk = sitk.ReadImage(str(file_path), pixel_type)

    # Do not deal with vector images here
    if image_sitk.GetNumberOfComponentsPerPixel() > 1:
        return image_sitk

    # Replace nan (and inf) with numerical values
    if replace_nan:
        image_nda = sitk.GetArrayFromImage(image_sitk)
        image_nda = np.nan_to_num(image_nda)
        image_sitk_ = sitk.GetImageFromArray(image_nda)
        image_sitk_.CopyInformation(image_sitk)
        return image_sitk_

    return image_sitk


##
# Print the ITK direction matrix
# \date       2016-09-20 15:52:28+0100
#
# \param      direction_itk  direction as obtained via image_itk.GetDirection()
#
def print_itk_direction(direction_itk):
    m_vnl = direction_itk.GetVnlMatrix()
    n_cols = m_vnl.cols()
    n_rows = m_vnl.rows()

    m_np = np.zeros((n_cols, n_rows))

    for i in range(0, n_cols):
        for j in range(0, n_rows):
            m_np[i, j] = m_vnl(i, j)

    print(m_np)


##
# Print the ITK array
# \date       2016-11-17 19:51:53+0000
#
# The itk-array can be instantiated via \p array_itk = \p itk.Array2D[itk.D]().
# Depending on the further use the size needs to be defined too, i.e. \p
# array_itk.SetSize(m,n)
#
# \param      array_itk  array as itk.Array2D#
# \f$ \in R^{m \times n}
# \f$ or parameter type of itk transforms
#
def print_itk_array(array_itk):

    # itk.Array2D[itk.D]()
    try:
        nda = get_numpy_from_itk_array(array_itk)

    # Parameter type
    except:
        N = array_itk.size()
        nda = np.zeros(N)

        # Fill array information
        for i in range(0, N):
            nda[i] = array_itk.GetElement(i)

    # Print array
    print(nda)


##
# Update ITK parameters
# \date       2017-06-26 17:04:43+0100
#
# \param      parameters_itk  The parameters itk
# \param      array           The array
#
def update_itk_parameters(parameters_itk, array):
    for i in range(0, array.size):
        parameters_itk.SetElement(i, array[i])


##
#       Gets the numpy from itk array.
# \date       2016-11-18 11:59:18+0000
#
# \todo That is VERY slow! needs improvement but no clue how to access the
# array quicker
#
# \param      array_itk  The array itk
#
# \return     The numpy from itk array.
#
def get_numpy_from_itk_array(array_itk):
    # time_start = ph.start_timing()
    cols = array_itk.cols()
    rows = array_itk.rows()
    nda = np.zeros((rows, cols))

    # Fill array information
    for i in range(0, rows):
        for j in range(0, cols):
            nda[i, j] = array_itk(i, j)

    # print(ph.stop_timing(time_start))

    # Even slower ...
    # time_start = ph.start_timing()
    # itk.GetArrayFromVnlMatrix(array_itk)
    # print(ph.stop_timing(time_start))

    return nda


##
# Gets the indices array to flattened sitk image data array.
# \date       2016-11-21 00:26:27+0000
#
# Get an (image_dimension x N_voxels)-numpy array which holds the indices
# corresponding to a sitk.GetArrayFromImage(image_sitk).flatten() data array
#
# \param      image_sitk  The image sitk
#
# \return     (image_dimension x N_voxels)-numpy array
#
def get_indices_array_to_flattened_sitk_image_data_array(image_sitk):

    shape = np.array(image_sitk.GetSize())[::-1]
    dim = image_sitk.GetDimension()

    x = np.arange(0, shape[0])
    y = np.arange(0, shape[1])
    if dim is 3:
        z = np.arange(0, shape[2])

    if dim is 2:
        # 'ij' yields vertical x-coordinate for image!
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Index array (2xN_voxels) of image in voxel space
        indices = np.array([Y.flatten(), X.flatten()])
    else:
        # 'ij' yields vertical x-coordinate for image!
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Index array (3xN_voxels) of image in voxel space
        indices = np.array([Z.flatten(), Y.flatten(), X.flatten()])

    return indices


##
# Gets the numpy array of jacobian itk transform applied on stack.
# \date          2016-11-18 12:00:05+0000
#
# Compute
# \f$ \frac{d\vec{T}}{d\vec{\theta}}(\vec{x}, \vec{\theta})
# \f$ for all points of the stack. The used parameter values
# \f$ \theta
# \f$ for evaluation are encoded in \p transform_itk.
#
# \param[in]     transform_itk                    Transform as itk object
# \param         image_sitk                       The image sitk
# \param[in]     points                           Grid points of image as
#                                                 (Dimension x N_voxel)-array.
# \param[in,out] jacobian_transform_on_image_nda  (N_voxel x transform_DOF)-
#                                                 array.
# \return        The (N_voxel x transform_DOF)-numpy array of the Jacobian of
#                the transform applied on the stack
#
def get_numpy_array_of_jacobian_itk_transform_applied_on_sitk_image(
        transform_itk,
        image_sitk,
        points=None,
        jacobian_transform_on_image_nda=None):

    # Shape of corresponding image data array
    shape = np.array(image_sitk.GetSize())
    dim = image_sitk.GetDimension()

    # Shall reduce the computational burden in case points stay the same.
    # However, it does not add much to the computational time
    if points is None:

        # Index array (dimension x N_voxels) of image in voxel space
        indices = get_indices_array_to_flattened_sitk_image_data_array(
            image_sitk)

        # Get transform from voxel to image space coordinates
        A = get_sitk_affine_matrix_from_sitk_image(
            image_sitk).reshape(dim, dim)
        t = np.array(image_sitk.GetOrigin()).reshape(dim, 1)

        # Compute point array (3xN_voxels) of image in image space
        points = A.dot(indices) + t

    if jacobian_transform_on_image_nda is None:
        # Allocate memory
        transform_dof = int(transform_itk.GetNumberOfParameters())
        jacobian_transform_on_image_nda = np.zeros(
            (points.shape[1], dim, transform_dof))

    # Create 2D itk-array
    jacobian_transform_on_point_itk = itk.Array2D[itk.D]()

    # Evaluate the Jacobian of transform at all points
    for i in range(0, points.shape[1]):

        # Compute Jacobian of transform w.r.t. parameters evaluated at point
        # jacobian_transform_point_itk is (Dimension x transform_DOF) array
        transform_itk.ComputeJacobianWithRespectToParameters(
            points[:, i], jacobian_transform_on_point_itk)

        # Convert itk to numpy array
        # THE computational time consuming part!
        jacobian_transform_on_image_nda[i, :, :] = get_numpy_from_itk_array(
            jacobian_transform_on_point_itk)

    # Return Jacobian w.r.t to parameters evaluated at all image points
    return jacobian_transform_on_image_nda


##
# Extract direction from SimpleITK-image so that it can be injected into
# ITK-image
# \date       2017-06-26 16:58:47+0100
#
# \param[in]  image_sitk  sitk.Image object
#
# \return     direction as itkMatrix object
#
def get_itk_direction_from_sitk_image(image_sitk):
    direction_sitk = image_sitk.GetDirection()

    return get_itk_from_sitk_direction(direction_sitk)


##
# Extract direction from ITK-image so that it can be injected into
# SimpleITK-image
# \date       2017-06-26 16:58:43+0100
#
# \param[in]  image_itk  itk.Image object
#
# \return     direction as 1D array of size dimension^2, np.array
#
def get_sitk_direction_from_itk_image(image_itk):
    direction_itk = image_itk.GetDirection()

    return get_sitk_from_itk_direction(direction_itk)


##
# Convert direction from sitk.Image to itk.Image direction format
# \date       2017-06-26 16:58:38+0100
#
# \param[in]  direction_sitk  direction obtained via GetDirection() of
#                             sitk.Image
#
# \return     direction which can be set as SetDirection() at itk.Image
#
def get_itk_from_sitk_direction(direction_sitk):
    dim = np.sqrt(len(direction_sitk)).astype('int')
    m = itk.vnl_matrix_fixed[itk.D, dim, dim]()

    for i in range(0, dim):
        for j in range(0, dim):
            m.set(i, j, direction_sitk[dim * i + j])

    return itk.Matrix[itk.D, dim, dim](m)


##
# Convert direction from itk.Image to sitk.Image direction format
# \date       2017-06-26 16:58:33+0100
#
# \param[in]  direction_itk  direction obtained via GetDirection() of itk.Image
#
# \return     direction which can be set as SetDirection() at sitk.Image
#
def get_sitk_from_itk_direction(direction_itk):
    vnl_matrix = direction_itk.GetVnlMatrix()
    dim = np.sqrt(vnl_matrix.size()).astype('int')

    direction_sitk = np.zeros(dim * dim)
    for i in range(0, dim):
        for j in range(0, dim):
            direction_sitk[i * dim + j] = vnl_matrix(i, j)

    return direction_sitk


##
# Convert itk.Euler3DTransform to sitk.Euler3DTransform instance
# \date       2017-06-26 16:58:27+0100
#
# \param[in]  Euler3DTransform_itk  itk.Euler3DTransform instance
#
# \return     converted sitk.Euler3DTransform instance
#
def get_sitk_from_itk_Euler3DTransform(Euler3DTransform_itk):
    parameters_itk = Euler3DTransform_itk.GetParameters()
    fixed_parameters_itk = Euler3DTransform_itk.GetFixedParameters()

    N_params = parameters_itk.GetNumberOfElements()
    N_fixedparams = fixed_parameters_itk.GetNumberOfElements()

    parameters_sitk = np.zeros(N_params)
    fixed_parameters_sitk = np.zeros(N_fixedparams)

    for i in range(0, N_params):
        parameters_sitk[i] = parameters_itk.GetElement(i)

    for i in range(0, N_fixedparams):
        fixed_parameters_sitk[i] = fixed_parameters_itk.GetElement(i)

    Euler3DTransform_sitk = sitk.Euler3DTransform()
    Euler3DTransform_sitk.SetParameters(parameters_sitk)
    Euler3DTransform_sitk.SetFixedParameters(fixed_parameters_sitk)

    return Euler3DTransform_sitk


def get_sitk_from_itk_transform(transform_itk):
    dimension = transform_itk.GetInputSpaceDimension()

    if transform_itk.GetNameOfClass() == "AffineTransform":
        transform_sitk = sitk.AffineTransform(dimension)
    else:
        transform_sitk = eval("sitk.%s()" % (transform_itk.GetNameOfClass()))

    parameters_itk = transform_itk.GetParameters()
    fixed_parameters_itk = transform_itk.GetFixedParameters()

    N_params = parameters_itk.GetNumberOfElements()
    N_fixedparams = fixed_parameters_itk.GetNumberOfElements()

    parameters_sitk = np.zeros(N_params)
    fixed_parameters_sitk = np.zeros(N_fixedparams)

    for i in range(0, N_params):
        parameters_sitk[i] = parameters_itk.GetElement(i)

    for i in range(0, N_fixedparams):
        fixed_parameters_sitk[i] = fixed_parameters_itk.GetElement(i)

    transform_sitk.SetParameters(parameters_sitk)
    transform_sitk.SetFixedParameters(fixed_parameters_sitk)

    return transform_sitk


def get_itk_from_sitk_transform(transform_sitk, pixel_type=itk.D):
    if transform_sitk.GetName() == "AffineTransform":
        transform_itk = itk.AffineTransform[
            pixel_type, transform_sitk.GetDimension()].New()
    else:
        transform_itk = getattr(itk, transform_sitk.GetName())[
            pixel_type].New()

    parameters_itk = transform_itk.GetParameters()
    parameters_sitk = transform_sitk.GetParameters()
    fixed_parameters_itk = transform_itk.GetFixedParameters()
    fixed_parameters_sitk = transform_sitk.GetFixedParameters()

    # Update transform parameters
    for i in range(len(parameters_sitk)):
        parameters_itk.SetElement(i, parameters_sitk[i])
    for i in range(len(fixed_parameters_sitk)):
        fixed_parameters_itk.SetElement(i, fixed_parameters_sitk[i])
    transform_itk.SetParameters(parameters_itk)
    transform_itk.SetFixedParameters(fixed_parameters_itk)

    return transform_itk


##
# Convert itk.AffineTransform to sitk.AffineTransform instance
# \date       2017-06-26 16:58:14+0100
#
# \param[in]  AffineTransform_itk  itk.AffineTransform instance
#
# \return     converted sitk.AffineTransform instance
#
def get_sitk_from_itk_AffineTransform(AffineTransform_itk):

    dim = len(AffineTransform_itk.GetTranslation())

    parameters_itk = AffineTransform_itk.GetParameters()
    fixed_parameters_itk = AffineTransform_itk.GetFixedParameters()

    N_params = parameters_itk.GetNumberOfElements()
    N_fixedparams = fixed_parameters_itk.GetNumberOfElements()

    parameters_sitk = np.zeros(N_params)
    fixed_parameters_sitk = np.zeros(N_fixedparams)

    for i in range(0, N_params):
        parameters_sitk[i] = parameters_itk.GetElement(i)

    for i in range(0, N_fixedparams):
        fixed_parameters_sitk[i] = fixed_parameters_itk.GetElement(i)

    AffineTransform_sitk = sitk.AffineTransform(dim)
    AffineTransform_sitk.SetParameters(parameters_sitk)
    AffineTransform_sitk.SetFixedParameters(fixed_parameters_sitk)

    return AffineTransform_sitk


##
# Convert SimpleITK-image to ITK-image
# \todo Check whether it is sufficient to just set origin, spacing and
# direction!
# \date       2017-06-26 16:58:03+0100
#
# \param[in]  image_sitk  SimpleITK-image to be converted, sitk.Image object
#
# \return     converted image as itk.Image object
#
def get_itk_from_sitk_image(image_sitk):

    # Extract information ready to use for ITK-image
    dimension = image_sitk.GetDimension()
    origin = image_sitk.GetOrigin()
    spacing = image_sitk.GetSpacing()
    direction = get_itk_direction_from_sitk_image(image_sitk)
    nda = sitk.GetArrayFromImage(image_sitk)

    # Define ITK image type according to pixel type of sitk.Object
    if image_sitk.GetPixelIDValue() is sitk.sitkFloat64:
        # image stack
        image_type = itk.Image[itk.D, dimension]
    else:
        # mask stack
        # Couldn't use itk.UC (which apparently is used for masks normally)
        # or any other "smaller format than itk.D" since
        # itk.MultiplyImageFilter[itk.UC, itk.D] does not work! (But
        # which I need within InverseProblemSolver)
        image_type = itk.Image[itk.D, dimension]
        nda = nda.astype(float)
        # image_type = itk.Image[itk.UI, dimension]
        # image_type = itk.Image[itk.UC, dimension]

    # Create ITK image
    itk2np = itk.PyBuffer[image_type]
    image_itk = itk2np.GetImageFromArray(nda)

    image_itk.SetOrigin(origin)
    image_itk.SetSpacing(spacing)
    image_itk.SetDirection(direction)

    image_itk.DisconnectPipeline()

    return image_itk


##
# Convert ITK-image to SimpleITK-image
# \todo Check whether it is sufficient to just set origin, spacing and
# direction!
# \date       2017-06-26 16:57:53+0100
#
# \param[in]  image_itk  ITK-image to be converted, itk.Image object
#
# \return     converted image as sitk.Image object
#
def get_sitk_from_itk_image(image_itk):

    # Extract information ready to use for SimpleITK-image
    dimension = image_itk.GetLargestPossibleRegion().GetImageDimension()
    origin = np.array(image_itk.GetOrigin())
    spacing = np.array(image_itk.GetSpacing())
    direction = get_sitk_direction_from_itk_image(image_itk)

    image_type = itk.Image[itk.D, dimension]
    itk2np = itk.PyBuffer[image_type]
    nda = itk2np.GetArrayFromImage(image_itk)

    # Create SimpleITK-image
    image_sitk = sitk.GetImageFromArray(nda)

    image_sitk.SetOrigin(origin)
    image_sitk.SetSpacing(spacing)
    image_sitk.SetDirection(direction)

    return image_sitk


##
# Print
# \date       2016-11-06 15:43:19+0000
#
# \param      rigid_affine_similarity_transform_sitk  The rigid affine
#                                                     similarity transform sitk
# \param      text                                    The text
#
def print_sitk_transform(rigid_affine_similarity_transform_sitk, text=None):

    dim = rigid_affine_similarity_transform_sitk.GetDimension()

    if text is None:
        text = rigid_affine_similarity_transform_sitk.GetName()

    matrix = np.array(
        rigid_affine_similarity_transform_sitk.GetMatrix()).reshape(dim, dim)
    translation = np.array(
        rigid_affine_similarity_transform_sitk.GetTranslation())

    parameters = np.array(
        rigid_affine_similarity_transform_sitk.GetParameters())
    center = np.array(
        rigid_affine_similarity_transform_sitk.GetFixedParameters())

    print("\t\t" + text + ":")
    print("\t\t\tcenter = " + str(center))

    if isinstance(rigid_affine_similarity_transform_sitk,
                  sitk.Euler3DTransform):
        print("\t\t\tangle_x, angle_y, angle_z = " +
              str(parameters[0:3]) + " rad")
        # print("\t\t\tangle_x, angle_y, angle_z = " +
        # str(parameters[0:3]*180/np.pi) + " deg")

    elif isinstance(rigid_affine_similarity_transform_sitk,
                    sitk.Euler2DTransform):
        print("\t\t\tangle = " + str(parameters[0]) + " rad")
        # print("\t\t\tangle = " + str(parameters[0]*180/np.pi) + " deg")

    elif isinstance(rigid_affine_similarity_transform_sitk,
                    sitk.Similarity2DTransform):
        print("\t\t\tangle = " + str(parameters[1]) + " rad")
        print("\t\t\tscale = " +
              str(rigid_affine_similarity_transform_sitk.GetScale()))

    elif isinstance(rigid_affine_similarity_transform_sitk,
                    sitk.Similarity3DTransform):
        print("\t\t\tangle_x, angle_y, angle_z = " +
              str(parameters[1:4]) + " rad")
        print("\t\t\tscale = " +
              str(rigid_affine_similarity_transform_sitk.GetScale()))

    print("\t\t\ttranslation = " + str(translation))

    # elif isinstance(rigid_affine_similarity_transform_sitk,
    # sitk.AffineTransform):
    print("\t\t\tmatrix = ")
    for i in range(0, dim):
        print("\t\t\t\t" + str(matrix[i, :]))


##
# Plot comparison between two 2D sitk.Image objects
# \date       2017-06-26 17:00:23+0100
#
# \param      image0_2D_sitk  The image 0 2d sitk
# \param      image1_2D_sitk  The image 1 2d sitk
# \param      fig_number      The fig number
# \param      flag_continue   The flag continue
#
def plot_compare_sitk_2D_images(image0_2D_sitk,
                                image1_2D_sitk,
                                fig_number=1,
                                flag_continue=0):

    fig = plt.figure(fig_number)
    plt.suptitle("intensity error norm = " +
                 str(np.linalg.norm(
                     sitk.GetArrayFromImage(image0_2D_sitk - image1_2D_sitk))))

    plt.subplot(1, 3, 1)
    plt.imshow(sitk.GetArrayFromImage(image0_2D_sitk), cmap="Greys_r")
    plt.title("image_0")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sitk.GetArrayFromImage(image1_2D_sitk), cmap="Greys_r")
    plt.title("image_1")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sitk.GetArrayFromImage(
        image0_2D_sitk - image1_2D_sitk), cmap="Greys_r")
    plt.title("image_0 - image_1")
    plt.axis('off')

    # Plot immediately or wait for following figures to come as well
    if flag_continue == 0:
        plt.show()
    else:
        # does not pause, but needs plt.show() at end
        plt.show(block=False)
        # of file to be visible
    return fig


##
# Plot slices of stack separately
# \date       2016-11-06 01:49:21+0000
#
# \param      stack_sitk  sitk.Image object to be plotted
# \param      cmap        Color map "Greys_r", "jet", etc.
# \param      title       The title
#
def plot_stack_of_slices(stack_sitk, cmap="Greys_r", title="slice"):
    nda = sitk.GetArrayFromImage(stack_sitk)
    ph.show_arrays(nda, cmap=cmap, title=title)


def plot_slices(slices, cmap="Greys_r", title="slice"):
    N = len(slices)
    nda0 = sitk.GetArrayFromImage(slices[0].sitk)

    nda = np.zeros((N, nda0.shape[0], nda0.shape[1]))
    nda[0, :, :] = nda0
    for i in range(1, N):
        nda[i, :, :] = sitk.GetArrayFromImage(slices[i].sitk)

    ph.show_arrays(nda, cmap=cmap, title=title)


def write_executable_file(cmds,
                          dir_output=DIR_TMP,
                          filename="show_comparison.py"):
    now = datetime.datetime.now()
    date_time = str(now.year) + "-" + str(now.month).zfill(2) + \
        "-" + str(now.day).zfill(2) + " "
    date_time += str(now.hour).zfill(2) + ":" + \
        str(now.minute).zfill(2) + ":" + str(now.second).zfill(2)

    dir_output_file = "./"

    # Substitute commands
    for i in range(0, len(cmds)):
        cmd = cmds[i]
        cmd = re.sub(dir_output + "/", '" directory + "', cmd)
        cmd = re.sub(ITKSNAP_EXE, 'ITKSNAP_EXE + "', cmd)
        cmd = re.sub(FSLVIEW_EXE, 'FSLVIEW_EXE + "', cmd)
        cmd = re.sub(NIFTYVIEW_EXE, 'NIFTYVIEW_EXE + "', cmd)
        cmds[i] = cmd

    call = "#!/usr/bin/python\n"
    call += "\n"
    call += "##\n"
    call += "#  \\file " + filename + "\n"
    call += "#  \\brief\t" + \
        "Execute '%s' for a visual comparison of all images\n" % filename
    call += "#\n"
    call += "#  Three different viewers can be selected:\n"
    call += "#  - ITK-SNAP (default): http://www.itksnap.org/\n"
    call += "#  - FSL: https://fsl.fmrib.ox.ac.uk/\n"
    call += "#  - NiftyView: http://cmictig.cs.ucl.ac.uk/research/software/software-nifty\n"
    call += "#\n"
    call += "#  \\author\t" + "Michael Ebner (michael.ebner.14@ucl.ac.uk)\n"
    call += "#  \\date\t" + date_time + "\n"
    call += "\n"
    call += "import os"
    call += "\n"
    call += "\n"
    call += "# Path to image data directory relative to this file:\n"
    call += "directory = " + '"' + dir_output_file + '"'
    call += "\n\n"
    call += "# Define executables:"
    call += "\n"
    call += "# ITKSNAP_EXE = " + '"/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP"'
    call += "\n"
    call += "ITKSNAP_EXE = " + '"' + ITKSNAP_EXE + '"'
    call += "\n"
    call += "# FSLVIEW_EXE = " + '"/usr/local/fsl/bin/fsleyes"'
    call += "\n"
    call += "FSLVIEW_EXE = " + '"' + FSLVIEW_EXE + '"'
    call += "\n"
    call += "# NIFTYVIEW_EXE = " + \
        '"/Applications/niftk-17.9.6/NiftyView.app/Contents/MacOS/NiftyView"'
    call += "\n"
    call += "NIFTYVIEW_EXE = " + '"' + NIFTYVIEW_EXE + '"'
    call += "\n"
    call += "\n"

    call += "# Define commands for respective viewers:"
    call += "\n"
    for i in range(0, len(cmds)):
        cmd = cmds[i]

        # for ITK-SNAP
        cmd = re.sub('\\\\\\n-o', '\\\\\\n" "-o', cmd)
        cmd = re.sub('\\\\\\n-s', '\\\\\\n" "-s', cmd)

        # put each image to new line
        if i == 0:
            cmd = re.sub('\\\\\\n"', '"\ncmd +=', cmd)
            cmd = re.sub('\\\\\\n&', '"\ncmd += "&', cmd)

        # same but add comment symbol
        else:
            cmd = re.sub('\\\\\\n"', '"\n# cmd +=', cmd)
            cmd = re.sub('\\\\\\n&', '"\n# cmd += "&', cmd)

        if i is 0:
            # Use first selected viewer
            call += "cmd = " + cmd + '" '
        else:
            # Comment the remaining viewers
            call += "# cmd = " + cmd + '" '
        call += "\n\n"

    call += "# Execute command to open selected viewer:"
    call += "\n"
    call += "print(cmd)"
    call += "\n"
    call += "os.system(cmd)\n"

    # Write function call to python file
    text_file = open(os.path.join(dir_output, filename), "w")
    text_file.write("%s" % call)
    text_file.close()

    ph.print_info("File " + os.path.join(dir_output, filename) + " generated.")

    # Make python file executable
    os.system("chmod +x " + os.path.join(dir_output, filename))


##
# Show image with ITK-Snap. Image is saved to /tmp/ for that purpose.
# \date       2016-09-19 16:47:18+0100
#
# \param      image_sitk            either single sitk.Image or list of
#                                   sitk.Images to overlay
# \param      label                 filename or list of filenames
# \param      segmentation          sitk.Image used as segmentation
# \param      show_comparison_file  choose whether comparison file shall be
#                                   produced to reproduce visualization at a
#                                   later stage
# \param      viewer                Can be "itksnap", "fsleyes", "NiftyView"
# \param      verbose               Show line for execution
# \param      interpolator          Interpolator used for resampling
# \param      dir_output            Output directory for writing files in order
#                                   to open them
# \param      default_pixel_value   default pixel value for interpolation,
#                                   float. Can also be "min" in case the
#                                   minimum data array value shall be used
#
def show_sitk_image(image_sitk,
                    label="test",
                    segmentation=None,
                    show_comparison_file=False,
                    name_comparison_file="show_comparison.py",
                    viewer=VIEWER,
                    verbose=True,
                    interpolator="Linear",
                    dir_output=DIR_TMP,
                    default_pixel_value=0):

    dir_output = ph.create_directory(dir_output)

    if viewer not in ["itksnap", "fsleyes", "NiftyView"]:
        raise ValueError(
            "Viewer not known. Select between 'itksnap', 'fsleyes' and 'NiftyView'")

    # Convert to list objects
    if type(image_sitk) is not list:
        image_sitk = [image_sitk]

    if type(label) is not list:
        label = [label]

    # Create a copy of the labels (string)
    label = list(label)

    label_segmentation = label[0] + "_seg"

    # Ensure label and image_sitk have same length
    if len(label) is not len(image_sitk):
        tmp = label
        label = [None] * len(image_sitk)
        for i in range(0, len(image_sitk)):
            label[i] = tmp[0] + str(i)

    # Write images to tmp-folder
    filenames = [None] * len(image_sitk)
    for i in range(0, len(image_sitk)):
        filenames[i] = os.path.join(dir_output, label[i] + ".nii.gz")
        write_nifti_image_sitk(image_sitk[i], filenames[i])

    if segmentation is None:
        filename_segmentation = None
    else:
        filename_segmentation = os.path.join(
            dir_output, label_segmentation + ".nii.gz")

        # In case images are not in the same physical space, resample them
        try:
            sitk.Cast(segmentation,
                      image_sitk[0].GetPixelIDValue()) - image_sitk[0]
        except:
            segmentation = sitk.Resample(
                segmentation,
                image_sitk[0],
                eval("sitk.Euler%dDTransform()" %
                     (image_sitk[0].GetDimension())),
                sitk.sitkNearestNeighbor,
                0)

        write_nifti_image_sitk(segmentation, filename_segmentation)

    # Get command line to call viewer
    cmd = eval("ph.get_function_call_" + viewer +
               "(filenames, filename_segmentation)")

    # Execute command
    ph.execute_command(cmd, verbose)

    # Create python script for the command above
    if show_comparison_file:

        cmds = [None] * 3
        ctr = 0
        cmds[ctr] = ph.get_function_call_itksnap(
            filenames, filename_segmentation)
        ctr = ctr + 1
        cmds[ctr] = ph.get_function_call_fsleyes(
            filenames, filename_segmentation)
        ctr = ctr + 1
        cmds[ctr] = ph.get_function_call_niftyview(
            filenames, filename_segmentation)
        ctr = ctr + 1

        # Build executable file containing the information
        write_executable_file(cmds, dir_output=dir_output,
                              filename=name_comparison_file)


##
# Visualize a list of Stack objects.
# \date       2016-11-05 23:18:23+0000
#
# \param      stacks                List of stack objects
# \param      label                 filename or list of filenames
# \param      segmentation          sitk.Image used as segmentation
# \param      show_comparison_file  choose whether comparison file shall be
#                                   produced to reproduce visualization at a
#                                   later stage
# \param      viewer                Can be "itksnap", "fsleyes", "NiftyView"
# \param      dir_output            Output directory for writing files in order
#                                   to open them
# \param      default_pixel_value   default pixel value for interpolation,
#                                   float. Can also be "min" in case the
#                                   minimum data array value shall be used
#
def show_stacks(stacks,
                label=None,
                segmentation=None,
                show_comparison_file=False,
                name_comparison_file="show_comparison.py",
                viewer=VIEWER,
                dir_output=DIR_TMP,
                default_pixel_value=0):

    N_stacks = len(stacks)
    images_sitk = [None] * N_stacks

    for i in range(0, N_stacks):
        images_sitk[i] = stacks[i].sitk

    if label is None:
        label = [None] * N_stacks
        for i in range(0, N_stacks):
            label[i] = stacks[i].get_filename()

    if segmentation is not None:
        segmentation_sitk = segmentation.sitk_mask

    else:
        segmentation_sitk = None

    show_sitk_image(images_sitk,
                    label=label,
                    segmentation=segmentation_sitk,
                    show_comparison_file=show_comparison_file,
                    name_comparison_file=name_comparison_file,
                    viewer=viewer,
                    dir_output=dir_output,
                    default_pixel_value=default_pixel_value)


##
# Show image with ITK-Snap. Image is saved to /tmp/ for that purpose
# \date       2017-06-26 17:01:41+0100
#
# \param[in]  image_itk     image to show as itk.object
# \param[in]  segmentation  as itk.image object
# \param[in]  overlay       image which shall be overlayed onto image_itk
#                           (optional)
# \param[in]  label         filename for file written to /tmp/ (optional)
# \param      dir_output    The dir output
#
def show_itk_image(image_itk,
                   segmentation=None,
                   overlay=None,
                   label="test",
                   dir_output=DIR_TMP,
                   ENDL=" "):

    if overlay is not None and segmentation is None:
        write_itk_image(image_itk,
                        os.path.join(dir_output, label + ".nii.gz"))
        write_itk_image(overlay,
                        os.path.join(dir_output, label + "_overlay.nii.gz"))

        cmd = ITKSNAP_EXE + " " \
            + "-g " + os.path.join(dir_output, label + ".nii.gz") + ENDL \
            + "-o " + os.path.join(dir_output,
                                   label + "_overlay.nii.gz") + ENDL \
            + "& "

    elif overlay is None and segmentation is not None:
        write_itk_image(image_itk,
                        os.path.join(dir_output, label + ".nii.gz"))
        write_itk_image(segmentation,
                        os.path.join(
                            dir_output, label + "_segmentation.nii.gz"))

        cmd = ITKSNAP_EXE + " " \
            + "-g " + os.path.join(dir_output, label + ".nii.gz") + ENDL \
            + "-s " + os.path.join(
                dir_output, label + "_segmentation.nii.gz") + ENDL \
            + "& "

    elif overlay is not None and segmentation is not None:
        write_itk_image(image_itk,
                        os.path.join(dir_output, label + ".nii.gz"))
        write_itk_image(segmentation,
                        os.path.join(dir_output,
                                     label + "_segmentation.nii.gz"))
        write_itk_image(overlay,
                        os.path.join(dir_output, label + "_overlay.nii.gz"))

        cmd = ITKSNAP_EXE + " " \
            + "-g " + os.path.join(dir_output,
                                   label + ".nii.gz") + ENDL \
            + "-s " + os.path.join(dir_output,
                                   label + "_segmentation.nii.gz") + ENDL \
            + "-o " + os.path.join(dir_output,
                                   label + "_overlay.nii.gz") + ENDL \
            + "& "

    else:
        write_itk_image(image_itk, os.path.join(dir_output, label + ".nii.gz"))

        cmd = ITKSNAP_EXE + " " \
            + "-g " + os.path.join(dir_output, label + ".nii.gz") + ENDL \
            + "& "

    ph.execute_command(cmd)


##
# Gets the downsampled sitk image.
# \date       2016-11-21 20:50:59+0000
#
# \param      image_sitk            The image sitk
# \param      downsampling_factors  The downsampling factors in x,y and z
#                                   direction as tuple
# \param      interpolator          The interpolator
# \param      default_pixel_value   The default pixel value
# \param      new_spacing           The new spacing in x, y and z as tuple
#
# \return     The downsampled sitk image.
#
def get_downsampled_sitk_image(image_sitk,
                               downsampling_factors=(1, 1, 1),
                               interpolator="NearestNeighbor",
                               default_pixel_value=0.0,
                               new_spacing=None):

    # Read required information from original image:
    dimension = image_sitk.GetDimension()
    spacing = np.array(image_sitk.GetSpacing()).astype('double')
    size = np.array(image_sitk.GetSize()).astype("int")

    if new_spacing is not None:
        downsampling_factors = new_spacing / spacing

    else:
        # Convert to array
        downsampling_factors = np.array(downsampling_factors).astype('double')

    # Choose interpolator
    try:
        interpolator_str = interpolator
        interpolator = eval("sitk.sitk" + interpolator_str)
    except:
        raise ValueError("Error: interpolator is not known")

    # Define new image space
    spacing_new = spacing * downsampling_factors[0:dimension]
    size_new = np.round(size / downsampling_factors[0:dimension]).astype("int")

    # Resampling
    image_sitk_resampled = sitk.Resample(
        image_sitk,
        size_new,
        eval("sitk.Euler" + str(dimension) + "DTransform()"),
        interpolator,
        image_sitk.GetOrigin(),
        spacing_new,
        image_sitk.GetDirection(),
        default_pixel_value,
        image_sitk.GetPixelIDValue())

    return image_sitk_resampled
