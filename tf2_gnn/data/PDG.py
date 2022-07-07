"""General dataset class for datasets with a numeric property stored as JSONLines files."""
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
import tensorflow as tf

from .graph_dataset import GraphBatchTFDataDescription, GraphDataset, GraphSample
from .jsonl_graph_dataset import JsonLGraphDataset

class GraphPDGSample(GraphSample):

    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_incoming_edges: np.ndarray,
        node_features: List[np.ndarray],
        Label: np.ndarray,
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_incoming_edges, node_features)
        self._Label = Label
    
    @property
    def Label(self) -> np.ndarray:
        """Node labels to predict as ndarray of shape [V, C]"""
        return self._Label

    def __str__(self):
        return (
            f"Adj:            {self._adjacency_lists}\n"
            f"Node_features:  {self._node_features}\n"
            f"Labels:  {self._Label}\n"
        )

GraphPDGSampleType = TypeVar(
    "GraphPDGSampleType", bound=GraphPDGSample
)

class  PDGDataset(JsonLGraphDataset[GraphPDGSample]):
    """
    Following the pattern of JsonLGraphPropertyDataset,
    but we need multiclass label support, not a property
    """ 
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        super_hypers = super().get_default_hyperparameters()
        this_hypers = {
            "num_fwd_edge_types": 2,
            "add_self_loop_edges": False,
            "tie_fwd_bkwd_edges": False,
        }
        super_hypers.update(this_hypers)
        return super_hypers
    
    def __init__(
        self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None, **kwargs,
    ):
        super().__init__(params, metadata=metadata, **kwargs)
        self._threshold_for_classification = params["threshold_for_classification"]

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    def _process_raw_datapoint(
        self, datapoint: Dict[str, Any]
    ) -> GraphPDGSampleType:
        node_features = datapoint["graph"]["node_features"]
        type_to_adj_list, type_to_num_incoming_edges = self._process_raw_adjacency_lists(
            raw_adjacency_lists=datapoint["graph"]["adjacency_lists"],
            num_nodes=len(node_features),
        )

        Label = datapoint["Label"]
        return GraphPDGSample(
            adjacency_lists=type_to_adj_list,
            type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
            node_features=node_features,
            Label=Label
        )

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["Label"] = []
        return new_batch

    def _add_graph_to_batch(
        self, raw_batch: Dict[str, Any], graph_sample: GraphPDGSampleType
    ) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["Label"].append(graph_sample.Label)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_labels = super()._finalise_batch(raw_batch)
        batch_labels["Label"] = np.concatenate(raw_batch["Label"], axis=0)
        return batch_features, batch_labels

    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        return GraphBatchTFDataDescription(
            batch_features_types=data_description.batch_features_types,
            batch_features_shapes=data_description.batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "Label": np.ndarray},
            batch_labels_shapes={**data_description.batch_labels_shapes, "Label": (None,None)},
        )