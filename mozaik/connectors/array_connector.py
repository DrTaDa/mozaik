# encoding: utf-8
"""
A pyNN connector that realizes a precomputed set of connections given as arrays.
"""
import numpy
from copy import deepcopy
from pyNN.connectors import Connector as PyNNConnector, FromListConnector
from pyNN.standardmodels import StandardSynapseType

# Maximum number of connections passed to a single nest.Connect call. Bounds the
# temporary memory used while converting the arrays.
_CONNECT_CHUNK = 2000000

_IGNORED_PARAMETERS = ('tau_minus', 'dendritic_delay_fraction', 'w_min_always_zero_in_NEST')


def _is_sorted_by_target_then_source(pre, post):
    if len(pre) < 2:
        return True
    dpost = numpy.diff(post)
    if numpy.any(dpost < 0):
        return False
    same_target = dpost == 0
    return not numpy.any(numpy.diff(pre)[same_target] <= 0)


class ArrayConnector(PyNNConnector):
    r"""
    Connects a projection according to a precomputed list of connections, given
    as four arrays (presynaptic indices, postsynaptic indices, weights, delays).

    It is semantically identical to pyNN's FromListConnector with columns
    (pre, post, weight, delay). With the NEST backend on a single MPI process
    all connections are created with a few array based nest.Connect calls instead
    of one nest.Connect call per target neuron. The connections are created in the
    same order as FromListConnector does (sorted by target, then by source), so
    NEST ends up with exactly the same connection tables.

    In all other cases it falls back to FromListConnector.
    """

    parameter_names = ()

    def __init__(self, pre, post, weights, delays):
        PyNNConnector.__init__(self, safe=False)
        self.pre = numpy.asarray(pre)
        self.post = numpy.asarray(post)
        self.weights = numpy.asarray(weights, dtype=numpy.float64)
        self.delays = numpy.asarray(delays, dtype=numpy.float64)

    def __len__(self):
        return len(self.pre)

    def as_list(self):
        return numpy.column_stack((self.pre, self.post, self.weights, self.delays))

    def connect(self, projection):
        if not self._bulk_connect_supported(projection):
            FromListConnector(self.as_list()).connect(projection)
            return
        if len(self.pre) == 0:
            return
        if numpy.any(self.pre >= projection.pre.size):
            from pyNN import errors
            raise errors.ConnectionError("source index out of range")

        import nest
        from pyNN.nest import simulator

        pre, post, weights, delays = self.pre, self.post, self.weights, self.delays
        if not _is_sorted_by_target_then_source(pre, post):
            order = numpy.lexsort((pre, post))
            pre, post, weights, delays = pre[order], post[order], weights[order], delays[order]
        if not numpy.all(projection.post._mask_local):
            local = projection.post._mask_local[post]
            pre, post, weights, delays = pre[local], post[local], weights[local], delays[local]

        # Same parameter handling as FromListConnector.connect and
        # pyNN.nest.Projection._convergent_connect.
        parameters = deepcopy(projection.synapse_type.parameter_space)
        if isinstance(projection.synapse_type, StandardSynapseType):
            parameters.shape = (len(pre),)
            parameters.update(weight=weights, delay=delays)
            parameters = projection.synapse_type.translate(parameters)
            parameters.evaluate(simplify=True)
            parameters = dict(parameters.items())
            weights = numpy.broadcast_to(numpy.asarray(parameters.pop('weight'), dtype=numpy.float64), (len(pre),))
            delays = numpy.broadcast_to(numpy.asarray(parameters.pop('delay'), dtype=numpy.float64), (len(pre),))
        else:
            # native synapse types are not translated
            parameters.shape = (1,)
            parameters.evaluate(simplify=True)
            parameters = dict(parameters.items())
            parameters.pop('weight', None)
            parameters.pop('delay', None)
        for name in _IGNORED_PARAMETERS:
            parameters.pop(name, None)

        weight_factor = 1.0
        if projection.receptor_type == 'inhibitory' and projection.post.conductance_based:
            weight_factor = -1.0
        if hasattr(projection.post, "celltype") and hasattr(projection.post.celltype, "receptor_scale"):
            weight_factor = weight_factor * projection.post.celltype.receptor_scale

        # Homogeneous parameters were already set as the defaults of the
        # projection's labelled NEST synapse model by pyNN (Projection.__init__),
        # so only inhomogeneous ones have to be passed explicitly.
        syn_params = {}
        for name, value in parameters.items():
            if isinstance(value, numpy.ndarray) and value.ndim > 0:
                syn_params[name] = numpy.broadcast_to(numpy.asarray(value, dtype=numpy.float64), (len(pre),))
        if not projection.post.celltype.standard_receptor_type:
            syn_params['receptor_type'] = projection.post.celltype.get_receptor_type(projection.receptor_type)

        nest.SetDefaults(projection.nest_synapse_model, {'synapse_label': projection.nest_synapse_label})
        pre_cells = numpy.asarray(projection.pre.all_cells, dtype=numpy.int64)
        post_cells = numpy.asarray(projection.post.all_cells, dtype=numpy.int64)
        for start in range(0, len(pre), _CONNECT_CHUNK):
            sl = slice(start, start + _CONNECT_CHUNK)
            w = weights[sl]
            if weight_factor != 1.0:
                w = w * weight_factor
            syn_spec = {
                'synapse_model': projection.nest_synapse_model,
                'weight': numpy.ascontiguousarray(w),
                'delay': numpy.ascontiguousarray(delays[sl]),
            }
            for name, value in syn_params.items():
                syn_spec[name] = numpy.ascontiguousarray(value[sl]) if isinstance(value, numpy.ndarray) else value
            nest.Connect(pre_cells[pre[sl]], post_cells[post[sl]], 'one_to_one', syn_spec)

        projection._sources.update(pre_cells[numpy.flatnonzero(numpy.bincount(pre, minlength=len(pre_cells)))].tolist())
        projection._connections = None
        simulator.state.stale_connection_cache = True
        if projection._common_synapse_property_names is None:
            projection._identify_common_synapse_properties()

    @staticmethod
    def _bulk_connect_supported(projection):
        try:
            import nest
            from pyNN.nest.projections import Projection as NestProjection
            from pyNN.nest.standardmodels.synapses import TsodyksMarkramSynapse
        except ImportError:
            return False
        if not isinstance(projection, NestProjection):
            return False
        if nest.NumProcesses() != 1:
            # array based nest.Connect is not available with MPI
            return False
        synapse_type = projection.synapse_type
        if "stdp" in projection.nest_synapse_model:
            return False
        if (getattr(synapse_type, 'weight_variable', None) or 'weight') != 'weight':
            return False
        if (getattr(synapse_type, 'delay_variable', None) or 'delay') != 'delay':
            return False
        if isinstance(synapse_type, TsodyksMarkramSynapse):
            # pyNN copies tau_psc from the postsynaptic cells in this case
            return False
        if not isinstance(synapse_type, StandardSynapseType):
            for name, value in synapse_type.parameter_space.items():
                if name not in ('weight', 'delay') and not value.is_homogeneous:
                    return False
        return True
