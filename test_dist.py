import sys
sys.path.insert(0, 'src')
from neuralmem.dist import ClusterCoordinator, MemorySharder, ReplicaManager, NodeDiscovery

# 1. Sharding
sharder = MemorySharder(virtual_nodes=100)
sharder.add_node('node-a')
sharder.add_node('node-b')
sharder.add_node('node-c')
print('=== Sharding ===')
print('Nodes:', sharder.get_nodes())
print('Ring size:', sharder.ring_size())
for k in ['k1','k2','k3','k4','k5']:
    print(f'  {k} -> {sharder.get_node(k)}')
print('Preference k1:', sharder.get_preference_list('k1', 3))

# 2. Replication
rm = ReplicaManager(default_replica_count=2)
primary = sharder.get_node('k1')
pl = sharder.get_preference_list('k1', 3)
print('=== Replication ===')
print('Assigned:', rm.assign_replicas('k1', primary, pl, 2))
rm.handle_node_failure(primary)
print('After primary failure:', rm.get_primary('k1'))

# 3. Discovery
disc = NodeDiscovery(heartbeat_timeout=3.0)
disc.start()
disc.register('node-x')
print('=== Discovery ===')
print('Nodes:', disc.get_nodes())
print('Alive:', disc.is_alive('node-x'))

# 4. Coordinator
sharder2 = MemorySharder(50)
rm2 = ReplicaManager(1)
disc2 = NodeDiscovery()
coord = ClusterCoordinator('master', sharder2, rm2, disc2, 50)
coord.start()
print('=== Coordinator ===')
print('Summary:', coord.cluster_summary())
print('Route foo:', coord.route_key('foo'))
coord.stop()
print('ALL PASS')
