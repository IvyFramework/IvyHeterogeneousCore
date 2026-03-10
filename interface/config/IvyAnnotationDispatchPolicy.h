#ifndef IVYANNOTATIONDISPATCHPOLICY_H
#define IVYANNOTATIONDISPATCHPOLICY_H

/**
 * @file IvyAnnotationDispatchPolicy.h
 * @brief Central annotation/dispatch policy for host-only graph paths vs host-device numeric paths.
 *
 * This policy is intentionally small and header-only so that call sites can adopt
 * a single, explicit qualifier vocabulary:
 * - graph-construction and pointer-gradient paths are host-only,
 * - pure numeric evaluation paths remain host-device eligible.
 *
 * The separation prevents accidental device-side instantiation of lazy graph code
 * while preserving the external autodiff API (e.g. func->gradient(x)).
 */

#include "config/IvyCudaFlags.h"

/** @brief Qualifier for pure numeric code paths that are valid on host and device. */
#define IVY_MATH_NUMERIC_QUALIFIER __HOST_DEVICE__

/** @brief Qualifier for host-only graph construction/evaluation paths. */
#define IVY_MATH_GRAPH_QUALIFIER __HOST__

/** @brief Qualifier for host-only tensor/STL evaluation paths. */
#define IVY_MATH_TENSOR_STL_QUALIFIER __HOST__

#endif
