"""Distributed storage 单元测试 — 全部使用 mock."""
from __future__ import annotations

from unittest.mock import MagicMock

from neuralmem.dist.sharding import MemorySharder
from neuralmem.dist.replication import ReplicaManager
from neuralmem.dist.discovery import NodeDiscovery
from neuralmem.dist.coordinator import ClusterCoordinator


# --------------------------------------------------------------------------- #
# MemorySharder
# --------------------------------------------------------------------------- #

def test_sharder_init():
    sharder = MemorySharder(virtual_nodes=10)
    assert sharder is not None


def test_sharder_add_node():
    sharder = MemorySharder(virtual_nodes=10)
    sharder.add_node("node1")
    assert "node1" in sharder.get_nodes()


def test_sharder_remove_node():
    sharder = MemorySharder(virtual_nodes=10)
    sharder.add_node("node1")
    sharder.remove_node("node1")
    assert "node1" not in sharder.get_nodes()


def test_sharder_route_key():
    sharder = MemorySharder(virtual_nodes=10)
    sharder.add_node("node1")
    sharder.add_node("node2")
    node = sharder.get_node("memory_123")
    assert node in ["node1", "node2"]


def test_sharder_preference_list():
    sharder = MemorySharder(virtual_nodes=10)
    sharder.add_node("node1")
    sharder.add_node("node2")
    sharder.add_node("node3")
    prefs = sharder.get_preference_list("memory_123", n=2)
    assert len(prefs) == 2
    assert len(set(prefs)) == 2  # Unique nodes


# --------------------------------------------------------------------------- #
# ReplicaManager
# --------------------------------------------------------------------------- #

def test_replica_manager_init():
    rm = ReplicaManager(default_replica_count=2)
    assert rm.default_replica_count == 2


def test_replica_manager_assign():
    rm = ReplicaManager(default_replica_count=2)
    rm.assign_replicas("memory_1", "node1", ["node2"])
    info = rm.get_primary("memory_1")
    assert info == "node1"


def test_replica_manager_failover():
    rm = ReplicaManager(default_replica_count=2)
    rm.assign_replicas("memory_1", "node1", ["node2"])
    rm.handle_node_failure("node1")
    # After primary failure, check if node2 is now primary
    info = rm.get_primary("memory_1")
    assert info == "node2"


# --------------------------------------------------------------------------- #
# NodeDiscovery
# --------------------------------------------------------------------------- #

def test_node_discovery_init():
    nd = NodeDiscovery(heartbeat_timeout=60)
    assert nd is not None


def test_node_discovery_register():
    nd = NodeDiscovery(heartbeat_timeout=60)
    nd.register("node1", metadata={"host": "localhost", "port": 8080})
    nodes = nd.get_nodes()
    assert any(n == "node1" for n in nodes)


def test_node_discovery_heartbeat():
    nd = NodeDiscovery(heartbeat_timeout=60)
    nd.register("node1", metadata={"host": "localhost", "port": 8080})
    nd.heartbeat("node1")
    assert nd.is_alive("node1")


def test_node_discovery_deregister():
    nd = NodeDiscovery(heartbeat_timeout=60)
    nd.register("node1", metadata={"host": "localhost", "port": 8080})
    nd.deregister("node1")
    assert not nd.is_alive("node1")


# --------------------------------------------------------------------------- #
# ClusterCoordinator
# --------------------------------------------------------------------------- #

def test_coordinator_init():
    sharder = MemorySharder(virtual_nodes=10)
    replica_mgr = ReplicaManager(default_replica_count=2)
    discovery = NodeDiscovery(heartbeat_timeout=60)
    cc = ClusterCoordinator("self", sharder, replica_mgr, discovery)
    assert cc.node_id == "self"


def test_coordinator_join_cluster():
    sharder = MemorySharder(virtual_nodes=10)
    replica_mgr = ReplicaManager(default_replica_count=2)
    discovery = NodeDiscovery(heartbeat_timeout=60)
    cc = ClusterCoordinator("self", sharder, replica_mgr, discovery)
    cc.start()
    assert "self" in discovery.get_nodes()


def test_coordinator_route_key():
    sharder = MemorySharder(virtual_nodes=10)
    replica_mgr = ReplicaManager(default_replica_count=2)
    discovery = NodeDiscovery(heartbeat_timeout=60)
    cc = ClusterCoordinator("self", sharder, replica_mgr, discovery)
    
    sharder.add_node("self")
    sharder.add_node("node2")
    
    node = cc.route_key("memory_123")
    assert node in ["self", "node2"]
